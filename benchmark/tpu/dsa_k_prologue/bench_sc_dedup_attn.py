"""Test only: attention over a gather chunk's distinct rows, without expanding them.

After the SparseCore dedup (bench_sc_dedup_mark.py) a chunk of 128 queries has D
distinct KV rows (D ~ 4-6% of 128 x 2048 in GLM-5.2) and each query uses 2048 of
them. Expanding the rows back per query (XLA take) costs ~1 ms, so here every
query attends to all D rows with its own selection as an additive mask:
flash-style, online softmax over blocks of TK distinct rows.

This does D / 2048 times the FLOPs of the sparse kernel (4-8x), so it wins only
if the TensorCore runs it in about the time the full gather takes now (~185 us
per chunk). Measured per point (one chunk, 128 queries x 64 heads, MLA 576/512):

* ``attn``: the Pallas kernel, ``qk`` = bf16 (Q bf16, K fp8 dequantized in the
  kernel) or fp8 (Q and K fp8 into the MXU), ``tk`` rows per KV block.
* ``mask``: building the per-query membership bitmap from the inverse index on
  the SparseCore (dsa_select_bits.select_bits), the kernel's other input.

Correctness on every point: output vs f32 sparse attention over each query's own
2048 rows (gathered explicitly). Prints ``ATTN`` JSON lines.
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

from sglang_tpu.attention.pallas.dsa_select_bits import (  # noqa: E402
    select_bits,
    select_bits_num_words,
)

Q = 128  # queries per chunk
H = 64  # heads (attention TP = 1)
DQK = 576  # absorbed latent 512 + rope 64
DV = 512
TOPK = 2048
TQ = 512  # query-head rows per tile = 8 queries x 64 heads
QPT = TQ // H
NEG = -1e30
F8 = jnp.float8_e4m3fn


def _kernel(q_ref, k_ref, b_ref, o_ref, m_sc, l_sc, acc_sc, *, nk, scale, fp8):
    j = pl.program_id(1)

    @pl.when(j == 0)
    def _():
        m_sc[...] = jnp.full(m_sc.shape, NEG, jnp.float32)
        l_sc[...] = jnp.zeros(l_sc.shape, jnp.float32)
        acc_sc[...] = jnp.zeros(acc_sc.shape, jnp.float32)

    k8 = k_ref[...]
    if fp8:
        q, k = q_ref[...], k8
    else:
        q, k = q_ref[...], k8.astype(jnp.bfloat16)
    s = lax.dot_general(q, k, (((1,), (1,)), ((), ())),
                        preferred_element_type=jnp.float32) * scale
    tk = s.shape[-1]
    s = (s.reshape(QPT, H, tk) + b_ref[...][:, None, :]).reshape(TQ, tk)
    m_prev = m_sc[...]
    m_new = jnp.maximum(m_prev, s.max(axis=-1, keepdims=True))
    p = jnp.exp(s - m_new)
    alpha = jnp.exp(m_prev - m_new)
    l_sc[...] = alpha * l_sc[...] + p.sum(axis=-1, keepdims=True)
    v = k8[:, :DV].astype(jnp.bfloat16)
    acc_sc[...] = alpha * acc_sc[...] + jnp.dot(
        p.astype(jnp.bfloat16), v, preferred_element_type=jnp.float32)
    m_sc[...] = m_new

    @pl.when(j == nk - 1)
    def _():
        o_ref[...] = (acc_sc[...] / l_sc[...]).astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=("tk", "fp8"))
def dense_masked_attn(q, k, bias, *, tk, fp8):
    """q bf16[Q*H, 576] (query-major, heads inner); k fp8[D, 576] distinct rows;
    bias f32[Q, D] (0 = selected, NEG = not). -> bf16[Q*H, 512]."""
    d = k.shape[0]
    nk = d // tk
    if fp8:
        q = q.astype(F8)
    return pl.pallas_call(
        functools.partial(_kernel, nk=nk, scale=DQK ** -0.5, fp8=fp8),
        grid=(Q * H // TQ, nk),
        in_specs=[
            pl.BlockSpec((TQ, DQK), lambda i, j: (i, 0)),
            pl.BlockSpec((tk, DQK), lambda i, j: (j, 0)),
            pl.BlockSpec((QPT, tk), lambda i, j: (i, j)),
        ],
        out_specs=pl.BlockSpec((TQ, DV), lambda i, j: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((Q * H, DV), jnp.bfloat16),
        scratch_shapes=[pltpu.VMEM((TQ, 1), jnp.float32),
                        pltpu.VMEM((TQ, 1), jnp.float32),
                        pltpu.VMEM((TQ, DV), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=48 * 1024 * 1024),
        name=f"dedup_dense_attn_d{d}_tk{tk}_{'fp8' if fp8 else 'bf16'}",
    )(q, k, bias)


@jax.jit
def sparse_reference(q, k, inv):
    """f32 attention of each query over its own 2048 rows k[inv[q]]."""
    kq = jnp.take(k.astype(jnp.float32), inv, axis=0)  # [Q, 2048, 576]
    qf = q.astype(jnp.float32).reshape(Q, H, DQK)
    s = jnp.einsum("qhd,qkd->qhk", qf, kq) * DQK ** -0.5
    p = jax.nn.softmax(s, axis=-1)
    return jnp.einsum("qhk,qkd->qhd", p, kq[..., :DV]).reshape(Q * H, DV)


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
    for d in (8192, 12288, 16384):
        inv = np.stack([rng.choice(d, TOPK, replace=False) for _ in range(Q)]).astype(np.int32)
        bias = np.full((Q, d), NEG, np.float32)
        bias[np.arange(Q)[:, None], inv] = 0.0
        q = jnp.asarray(rng.normal(0, 1, (Q * H, DQK)), jnp.bfloat16)
        k = jnp.asarray(np.clip(rng.normal(0, 1, (d, DQK)), -448, 448), F8)
        inv_j, bias_j = jnp.asarray(inv), jnp.asarray(bias)
        ref = np.asarray(sparse_reference(q, k, inv_j), np.float32)
        ref_norm = float(np.abs(ref).max())

        nw = select_bits_num_words(d, 64)
        row = dict(part="mask", d=d, num_words=nw)
        try:
            row["us"], row["us_range"] = timed(lambda: select_bits(inv_j, num_words=nw))
        except Exception as e:  # report, do not hide
            row["error"] = repr(e)[:500]
        print("ATTN " + json.dumps(row), flush=True)

        flop = 2 * Q * H * d * (DQK + DV)
        for fp8 in (False, True):
            for tk in (512, 1024):
                row = dict(part="attn", d=d, qk="fp8" if fp8 else "bf16", tk=tk,
                           flop=flop, sparse_flop=2 * Q * H * TOPK * (DQK + DV))
                try:
                    f = lambda: dense_masked_attn(q, k, bias_j, tk=tk, fp8=fp8)  # noqa: E731
                    out = np.asarray(f(), np.float32)
                    row["max_err_rel"] = round(float(np.abs(out - ref).max()) / ref_norm, 4)
                    row["us"], row["us_range"] = timed(f)
                    row["tflops"] = round(flop / row["us"] / 1e6, 1)
                except Exception as e:  # report, do not hide
                    row["error"] = repr(e)[:500]
                print("ATTN " + json.dumps(row), flush=True)
    print("ATTN_DONE", flush=True)


if __name__ == "__main__":
    main()
