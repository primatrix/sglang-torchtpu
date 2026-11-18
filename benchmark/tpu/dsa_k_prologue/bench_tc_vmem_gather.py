"""Test only: how fast the TensorCore gathers rows that already sit in VMEM.

The dedup plan (attention_sparsecore_layout.md) keeps a chunk's D distinct KV rows
(D ~ 4-16K in GLM-5.2) in VMEM and has the attention kernel read each query's 2048
rows by its inverse index. That pays only if the TensorCore reads 128 x 2048 =
262,144 rows per chunk well below the native SparseCore HBM gather (~180 us per
chunk, 0.547 ns/row). Budget: SC dedup (~17 us/chunk) + SC gather of the distinct
rows (~65 us) + attention (~23 us) leaves ~110 us, i.e. < ~0.42 ns/row.

One NoPE row is 128 x int32 (512 B, one lane row). Grid over blocks of QB queries;
each query's 2048 indices come in SMEM; rows go to a VMEM buffer that is then
summed (cheap, keeps the gather live). Variants:

* ``vld``: one dynamic-sublane load + store per row (``k[pl.ds(i, 1)]``).
* ``dma_vmem``: one local VMEM -> VMEM async copy per row, all in flight, then waits.
* ``dma_hbm``: K in HBM, one HBM -> VMEM async copy per row (TensorCore gather
  without dedup, for reference).
* ``contig``: the same 2048 rows as one contiguous slice (no gather; the floor).

Prints ``VGATH`` JSON lines: us per chunk, ns per row, and an exact check of the
summed rows against numpy.
"""

import functools
import json
import os
import statistics
import time

os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402

Q = 128
TOPK = 2048
W = 128  # int32 words per row (512 B NoPE)


def _reduce(buf, o_ref, qi):
    o_ref[qi] = buf[...].reshape(TOPK // 8, 8, W).sum(axis=0)


def _vld_kernel(idx_ref, k_ref, o_ref, buf, *, qb, unroll):
    for qi in range(qb):
        def body(r, c, qi=qi):
            buf[pl.ds(r, 1), :] = k_ref[pl.ds(idx_ref[qi * TOPK + r], 1), :]
            return c
        lax.fori_loop(0, TOPK, body, 0, unroll=unroll)
        _reduce(buf, o_ref, qi)


def _dma_kernel(idx_ref, k_ref, o_ref, buf, sem, *, qb):
    for qi in range(qb):
        def start(r, c, qi=qi):
            pltpu.make_async_copy(k_ref.at[pl.ds(idx_ref[qi * TOPK + r], 1)],
                                  buf.at[pl.ds(r, 1)], sem).start()
            return c
        lax.fori_loop(0, TOPK, start, 0, unroll=8)

        def wait(r, c):
            pltpu.make_async_copy(k_ref.at[pl.ds(0, 1)], buf.at[pl.ds(r, 1)], sem).wait()
            return c
        lax.fori_loop(0, TOPK, wait, 0, unroll=8)
        _reduce(buf, o_ref, qi)


def _iar_kernel(idx_ref, k_ref, o_ref, buf, *, qb, acc):
    """Vector-index gather: 8 rows per load from an (8,) index vector, which
    Mosaic lowers to vsetiar.sublane + vld.iar (TensorCore ISA topic 15).
    idx_ref: VMEM i32[qb * 16, 128], 16 rows of 128 indices per query.
    ``acc``: sum the rows in registers (what a fused QK would consume) instead
    of storing them to the [2048, 128] buffer."""
    for qi in range(qb):
        def body(rw, a, qi=qi):
            irow = idx_ref[pl.ds(qi * 16 + rw, 1), :]  # (1, 128)
            for j in range(16):
                v = irow[0, j * 8:(j + 1) * 8]
                rows = k_ref[v, :]  # (8, 128)
                if acc:
                    a = a + rows
                else:
                    buf[pl.ds(pl.multiple_of(rw * 128 + j * 8, 8), 8), :] = rows
            return a
        a = lax.fori_loop(0, TOPK // 128, body, jnp.zeros((8, W), jnp.int32))
        if acc:
            o_ref[qi] = a
        else:
            _reduce(buf, o_ref, qi)


# Test only: Pallas rejects vector indices into a TensorCore ref ("Cannot do int
# indexing on TPU") and uses tpu.vector_load_idx only for SparseCore load_gather.
# This private primitive emits tpu.vector_load_idx in a TensorCore kernel, to see
# whether Mosaic accepts it there and lowers it to vsetiar + vld.iar.
from jax._src import core as _jax_core  # noqa: E402
from jax._src.lib.mlir import ir as _ir  # noqa: E402
from jax._src.pallas.mosaic import lowering as _tcl  # noqa: E402
from jax._src.state import types as _state_types  # noqa: E402
from jax.experimental.mosaic.dialects import tpu as _tpu_d  # noqa: E402

tc_load_idx_p = _jax_core.Primitive("tc_vector_load_idx")


@tc_load_idx_p.def_effectful_abstract_eval
def _tc_load_idx_abstract(ref, *idx):
    return _jax_core.ShapedArray(idx[0].shape, ref.dtype), {_state_types.ReadEffect(0)}


@_tcl.register_lowering_rule(tc_load_idx_p)
def _tc_load_idx_lowering(ctx, ref, *idx):
    out = ctx.avals_out[0]
    vt = _ir.VectorType.get(out.shape, _tcl._dtype_to_ir_type(out.dtype))
    return _tpu_d.vector_load_idx(vt, ref, list(idx))


def _iarx_kernel(idx_ref, k_ref, o_ref, buf, *, qb, acc, form):
    """idx_ref: i32[qb * 8, 256]; column c of a query's (8, 256) block holds its
    rows 8c..8c+7 (one per sublane), so a static column slice broadcast along the
    lanes is the per-sublane row index (no lane-to-sublane relayout).
    form "2d": indices (row, lane) per element; "1d": the row index only."""
    lane = lax.broadcasted_iota(jnp.int32, (8, W), 1)
    for qi in range(qb):
        def body(cb, a, qi=qi):
            blk = idx_ref[pl.ds(qi * 8, 8), pl.ds(pl.multiple_of(cb * 128, 128), 128)]
            for j in range(128):
                r = jnp.broadcast_to(blk[:, j:j + 1], (8, W))
                rows = tc_load_idx_p.bind(k_ref, *((r, lane) if form == "2d" else (r,)))
                if acc:
                    a = a + rows
                else:
                    buf[pl.ds(pl.multiple_of(cb * 1024 + j * 8, 8), 8), :] = rows
            return a
        a = lax.fori_loop(0, TOPK // 1024, body, jnp.zeros((8, W), jnp.int32))
        if acc:
            o_ref[qi] = a
        else:
            _reduce(buf, o_ref, qi)


def _contig_kernel(idx_ref, k_ref, o_ref, buf, *, qb):
    for qi in range(qb):
        s = pl.multiple_of(jnp.minimum(idx_ref[qi * TOPK], k_ref.shape[0] - TOPK) // 8 * 8, 8)
        buf[...] = k_ref[pl.ds(s, TOPK), :]
        _reduce(buf, o_ref, qi)


@functools.partial(jax.jit, static_argnames=("variant", "qb", "unroll"))
def gather(k, idx, *, variant, qb, unroll=8):
    d = k.shape[0]
    # 1D blocks of qb * 2048 indices: a (qb, 2048) SMEM block with qb < 8 fails
    # the TPU block-shape rule (last two dims divisible by 8 and 128).
    idx_spec = pl.BlockSpec((qb * TOPK,), lambda i: (i,), memory_space=pltpu.SMEM)
    if variant.startswith("iarx"):
        idx_spec = pl.BlockSpec((qb * 8, 256), lambda i: (i, 0))
        idx = idx.reshape(Q, 256, 8).transpose(0, 2, 1).reshape(Q * 8, 256)
    elif variant.startswith("iar"):
        idx_spec = pl.BlockSpec((qb * 16, 128), lambda i: (i, 0))
        idx = idx.reshape(-1, 128)
    if variant == "dma_hbm":
        k_spec = pl.BlockSpec(memory_space=pl.ANY)
    else:
        k_spec = pl.BlockSpec((d, W), lambda i: (0, 0))
    scratch = [pltpu.VMEM((TOPK, W), jnp.int32)]
    if variant == "vld":
        kern = functools.partial(_vld_kernel, qb=qb, unroll=unroll)
    elif variant.startswith("iarx"):
        kern = functools.partial(_iarx_kernel, qb=qb, acc=variant.endswith("_acc"),
                                 form="1d" if "_1d" in variant else "2d")
    elif variant.startswith("iar"):
        kern = functools.partial(_iar_kernel, qb=qb, acc=variant == "iar_acc")
    elif variant in ("dma_vmem", "dma_hbm"):
        kern = functools.partial(_dma_kernel, qb=qb)
        scratch.append(pltpu.SemaphoreType.DMA(()))
    else:
        kern = functools.partial(_contig_kernel, qb=qb)
    return pl.pallas_call(
        kern,
        grid=(Q // qb,),
        in_specs=[idx_spec, k_spec],
        out_specs=pl.BlockSpec((qb, 8, W), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct((Q, 8, W), jnp.int32),
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",), vmem_limit_bytes=56 * 1024 * 1024),
        name=f"tc_vmem_gather_{variant}_d{d}_qb{qb}",
    )(idx if variant.startswith("iar") else idx.reshape(-1), k)


def reference(k, idx):
    rows = k[idx]  # [Q, 2048, W]
    return rows.reshape(Q, TOPK // 8, 8, W).sum(axis=1, dtype=np.int32)


def timed(fn, iters=20, repeats=5):
    jax.block_until_ready(fn())
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn()
        jax.block_until_ready(out)
        runs.append((time.perf_counter() - t0) / iters * 1e6)
    return round(statistics.median(runs), 1), [round(min(runs), 1), round(max(runs), 1)]


def main():
    rng = np.random.default_rng(0)
    rows = Q * TOPK
    for d in (4096, 8192, 16384):
        k_np = rng.integers(0, 1000, (d, W), dtype=np.int32)
        idx_np = np.stack([rng.choice(d, TOPK, replace=False) for _ in range(Q)]).astype(np.int32)
        k, idx = jnp.asarray(k_np), jnp.asarray(idx_np)
        ref = reference(k_np, idx_np)
        points = [("contig", 4, 8), ("vld", 4, 16), ("iarx_2d", 1, 0), ("iarx_2d", 4, 0),
                  ("iarx_2d_acc", 4, 0), ("iarx_1d", 4, 0), ("iarx_1d_acc", 4, 0)]
        for variant, qb, unroll in points:
            row = dict(d=d, variant=variant, qb=qb, unroll=unroll, rows=rows)
            try:
                f = lambda: gather(k, idx, variant=variant, qb=qb, unroll=unroll)  # noqa: E731
                out = np.asarray(f())
                row["exact"] = None if variant == "contig" else bool(np.array_equal(out, ref))
                row["us"], row["us_range"] = timed(f)
                row["ns_per_row"] = round(row["us"] * 1e3 / rows, 3)
            except Exception as e:  # report, do not hide
                row["error"] = repr(e)[:600]
            print("VGATH " + json.dumps(row), flush=True)
    print("VGATH_DONE", flush=True)


if __name__ == "__main__":
    main()
