"""Test only: dense masked attention over the distinct rows of small query groups.

Follow-up to bench_sc_dedup_attn.py. Deduping a whole 128-query chunk leaves
~9.3K distinct rows (4.5x the sparse FLOPs); smaller groups leave fewer. Real
GLM-5.2 top-k (exp-s2a0c6e4ss, layers 0/10/20): mean distinct rows per group of
g queries is 3871 (g=8), 4674 (g=16), 5803 (g=32).

One chunk = 128 queries = 128 / g groups. Group i has its own D_g distinct rows
(padded to a multiple of tk, pad rows masked) and each query picks 2048 of them.
Grid (group, kv block); the query tile is g queries x 64 heads. Reported per
point: kernel us per chunk, achieved TFLOPS on the dense FLOPs, and the error vs
f32 sparse attention over each query's own rows. Compare against today's
gather + attention, ~220 us per chunk (3.49 ms / 16). Prints ``GATTN`` lines.
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
H = 64
DQK = 576
DV = 512
TOPK = 2048
NEG = -1e30
F8 = jnp.float8_e4m3fn
MEAN_DISTINCT = {8: 3871, 16: 4674, 32: 5803}


def _kernel(q_ref, k_ref, b_ref, o_ref, m_sc, l_sc, acc_sc, *, nk, g, scale, fp8):
    j = pl.program_id(1)

    @pl.when(j == 0)
    def _():
        m_sc[...] = jnp.full(m_sc.shape, NEG, jnp.float32)
        l_sc[...] = jnp.zeros(l_sc.shape, jnp.float32)
        acc_sc[...] = jnp.zeros(acc_sc.shape, jnp.float32)

    k8 = k_ref[0]
    q = q_ref[...]
    k = k8 if fp8 else k8.astype(jnp.bfloat16)
    s = lax.dot_general(q, k, (((1,), (1,)), ((), ())),
                        preferred_element_type=jnp.float32) * scale
    tk = s.shape[-1]
    s = (s.reshape(g, H, tk) + b_ref[0][:, None, :]).reshape(g * H, tk)
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


@functools.partial(jax.jit, static_argnames=("g", "tk", "fp8"))
def grouped_attn(q, k, bias, *, g, tk, fp8):
    """q bf16[Q*H, 576]; k fp8[G, Dp, 576] per-group distinct rows; bias
    f32[G, g, Dp] (0 = selected). -> bf16[Q*H, 512]."""
    groups, dp, _ = k.shape
    nk = dp // tk
    if fp8:
        q = q.astype(F8)
    tq = g * H
    return pl.pallas_call(
        functools.partial(_kernel, nk=nk, g=g, scale=DQK ** -0.5, fp8=fp8),
        grid=(groups, nk),
        in_specs=[
            pl.BlockSpec((tq, DQK), lambda i, j: (i, 0)),
            pl.BlockSpec((1, tk, DQK), lambda i, j: (i, j, 0)),
            pl.BlockSpec((1, g, tk), lambda i, j: (i, 0, j)),
        ],
        out_specs=pl.BlockSpec((tq, DV), lambda i, j: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((Q * H, DV), jnp.bfloat16),
        scratch_shapes=[pltpu.VMEM((tq, 1), jnp.float32),
                        pltpu.VMEM((tq, 1), jnp.float32),
                        pltpu.VMEM((tq, DV), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=56 * 1024 * 1024),
        name=f"dedup_group_attn_g{g}_d{dp}_tk{tk}_{'fp8' if fp8 else 'bf16'}",
    )(q, k, bias)


@functools.partial(jax.jit, static_argnames=("g",))
def sparse_reference(q, k, inv, *, g):
    """f32 attention of each query over its own rows k[group, inv[q]]."""
    groups = Q // g
    kf = k.astype(jnp.float32)
    gid = jnp.repeat(jnp.arange(groups), g)  # [Q]
    kq = kf[gid[:, None], inv]  # [Q, 2048, 576]
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
    q = jnp.asarray(rng.normal(0, 1, (Q * H, DQK)), jnp.bfloat16)
    for g, d in MEAN_DISTINCT.items():
        groups = Q // g
        for tk in (256, 512):
            dp = -(-d // tk) * tk
            inv = np.stack([rng.choice(d, TOPK, replace=False) for _ in range(Q)]).astype(np.int32)
            bias = np.full((groups, g, dp), NEG, np.float32)
            for qi in range(Q):
                bias[qi // g, qi % g, inv[qi]] = 0.0
            k = jnp.asarray(np.clip(rng.normal(0, 1, (groups, dp, DQK)), -448, 448), F8)
            inv_j, bias_j = jnp.asarray(inv), jnp.asarray(bias)
            ref = np.asarray(sparse_reference(q, k, inv_j, g=g), np.float32)
            ref_norm = float(np.abs(ref).max())
            flop = 2 * Q * H * dp * (DQK + DV)
            for fp8 in (False, True):
                row = dict(g=g, distinct=d, d_pad=dp, tk=tk, qk="fp8" if fp8 else "bf16",
                           dense_x=round(dp / TOPK, 2), flop=flop)
                try:
                    f = lambda: grouped_attn(q, k, bias_j, g=g, tk=tk, fp8=fp8)  # noqa: E731
                    out = np.asarray(f(), np.float32)
                    row["max_err_rel"] = round(float(np.abs(out - ref).max()) / ref_norm, 4)
                    row["us"], row["us_range"] = timed(f)
                    row["tflops"] = round(flop / row["us"] / 1e6, 1)
                except Exception as e:  # report, do not hide
                    row["error"] = repr(e)[:500]
                print("GATTN " + json.dumps(row), flush=True)
    print("GATTN_DONE", flush=True)


if __name__ == "__main__":
    main()
