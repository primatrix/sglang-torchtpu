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
            buf[pl.ds(r, 1), :] = k_ref[pl.ds(idx_ref[qi, r], 1), :]
            return c
        lax.fori_loop(0, TOPK, body, 0, unroll=unroll)
        _reduce(buf, o_ref, qi)


def _dma_kernel(idx_ref, k_ref, o_ref, buf, sem, *, qb):
    for qi in range(qb):
        def start(r, c, qi=qi):
            pltpu.make_async_copy(k_ref.at[pl.ds(idx_ref[qi, r], 1)],
                                  buf.at[pl.ds(r, 1)], sem).start()
            return c
        lax.fori_loop(0, TOPK, start, 0, unroll=8)

        def wait(r, c):
            pltpu.make_async_copy(k_ref.at[pl.ds(0, 1)], buf.at[pl.ds(r, 1)], sem).wait()
            return c
        lax.fori_loop(0, TOPK, wait, 0, unroll=8)
        _reduce(buf, o_ref, qi)


def _contig_kernel(idx_ref, k_ref, o_ref, buf, *, qb):
    for qi in range(qb):
        s = pl.multiple_of(jnp.minimum(idx_ref[qi, 0], k_ref.shape[0] - TOPK) // 8 * 8, 8)
        buf[...] = k_ref[pl.ds(s, TOPK), :]
        _reduce(buf, o_ref, qi)


@functools.partial(jax.jit, static_argnames=("variant", "qb", "unroll"))
def gather(k, idx, *, variant, qb, unroll=8):
    d = k.shape[0]
    idx_spec = pl.BlockSpec((qb, TOPK), lambda i: (i, 0), memory_space=pltpu.SMEM)
    if variant == "dma_hbm":
        k_spec = pl.BlockSpec(memory_space=pl.ANY)
    else:
        k_spec = pl.BlockSpec((d, W), lambda i: (0, 0))
    scratch = [pltpu.VMEM((TOPK, W), jnp.int32)]
    if variant == "vld":
        kern = functools.partial(_vld_kernel, qb=qb, unroll=unroll)
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
    )(idx, k)


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
        points = [("contig", 4, 8), ("vld", 1, 8), ("vld", 4, 8), ("vld", 4, 16),
                  ("dma_vmem", 4, 8), ("dma_hbm", 4, 8)]
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
