"""Top-k selection shortfall of the target chain and of the fused kernel.

Both are scored against an exact reference: the f64 projection followed by the
target's own bf16 rounding of LayerNorm and rope, quantized to the index-key
record. For one query the shortfall is ``1 - (lowest exact score among its
top-k picks) / (exact k-th best score)``; 0 means the picks are a true top-k.

* ``target``: the torch_tpu chain reproduced in JAX, one jit per stage with the
  values pulled to the host in between. That form matched torch_tpu bit for bit
  on v7x (projection, LayerNorm, rope, x_scale).
* ``kernel``: ``key_prologue_pallas`` (Mosaic, native FP8 dot).

The kernel's acceptance bound is derived from the target's measured shortfall.
"""

import argparse
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax import lax

T, HIDDEN, D, ROPE, EPS, PAGE = 640, 6144, 128, 64, 1e-6, 128
TOPK, HEADS, QUERIES = 128, 32, 64
BF16, FP8 = ml_dtypes.bfloat16, ml_dtypes.float8_e4m3fn
f32 = jnp.float32


def _inputs(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (T, HIDDEN)).astype(BF16)
    wk = rng.normal(0, 0.02, (HIDDEN, D)).astype(FP8)
    ws = rng.uniform(0.5, 1.5, D).astype(np.float32)
    nw = rng.uniform(0.5, 1.5, D).astype(np.float32)
    nb = rng.normal(0, 0.1, D).astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi, (T, ROPE // 2))
    return x, wk, ws, nw, nb, np.cos(ang).astype(BF16), np.sin(ang).astype(BF16)


def _rt(a):
    return lax.reduce_precision(a.astype(f32), exponent_bits=8, mantissa_bits=7)


@jax.jit
def _proj(x, wk, ws):
    xf = x.astype(f32)
    xs = jnp.maximum(jnp.max(jnp.abs(xf), axis=1, keepdims=True) / 448.0, 1e-12)
    xq = (xf / xs).astype(jnp.float8_e4m3fn)
    acc = lax.dot_general(xq, wk, (((1,), (0,)), ((), ())), preferred_element_type=f32)
    return _rt(acc * xs * ws[None])


@jax.jit
def _norm(k, nw, nb):
    mean = jnp.mean(k, axis=1, keepdims=True)
    var = jnp.mean(jnp.square(k - mean), axis=1, keepdims=True)
    return _rt((k - mean) * lax.rsqrt(var + EPS) * nw + nb)


@jax.jit
def _rope(k, cos, sin):
    c, s = cos.astype(f32), sin.astype(f32)
    x1, x2 = k[:, 0:ROPE:2], k[:, 1:ROPE:2]
    o1 = _rt(_rt(x1 * c) - _rt(x2 * s))
    o2 = _rt(_rt(x2 * c) + _rt(x1 * s))
    rot = jnp.stack([o1, o2], axis=-1).reshape(k.shape[0], ROPE)
    return jnp.concatenate([rot, k[:, ROPE:]], axis=1)


def _target_keys(x, wk, ws, nw, nb, cos, sin):
    p = np.asarray(_proj(jnp.asarray(x), jnp.asarray(wk), jnp.asarray(ws)))
    n = np.asarray(_norm(jnp.asarray(p), jnp.asarray(nw), jnp.asarray(nb)))
    return np.asarray(_rope(jnp.asarray(n), jnp.asarray(cos), jnp.asarray(sin)))


def _bf16(a):
    return np.asarray(a, np.float32).astype(BF16).astype(np.float32)


def _exact_keys(x, wk, ws, nw, nb, cos, sin):
    x = x.astype(np.float32)
    xs = np.maximum(np.abs(x).max(1, keepdims=True) * (np.float32(1) / np.float32(448.0)), 1e-12)
    xq = (x / xs).astype(FP8).astype(np.float64)
    acc = (xq @ wk.astype(np.float32).astype(np.float64)).astype(np.float32)
    k = _bf16(acc * xs * ws[None])
    mu = k.mean(1, keepdims=True)
    var = ((k - mu) ** 2).mean(1, keepdims=True)
    k = _bf16((k - mu) / np.sqrt(var + EPS) * nw + nb)
    x1, x2 = k[:, 0:ROPE:2], k[:, 1:ROPE:2]
    c, s = _bf16(cos), _bf16(sin)
    k = k.copy()
    k[:, 0:ROPE:2] = _bf16(_bf16(x1 * c) - _bf16(x2 * s))
    k[:, 1:ROPE:2] = _bf16(_bf16(x2 * c) + _bf16(x1 * s))
    return k


def _quantized(k):
    """Dequantized index-key records of bf16-valued keys (quantize_fp8_ue8m0)."""
    from sglang_tpu.attention.pallas.util import quantize_fp8_ue8m0

    vals, scale, _ = quantize_fp8_ue8m0(jnp.asarray(k.astype(BF16)))
    return np.asarray(vals.astype(f32)) * np.asarray(scale)


def _kernel_keys(x, wk, ws, nw, nb, cos, sin):
    from sglang_tpu.attention.pallas.dsa_key_prologue import key_prologue_pallas

    pages = T // PAGE
    cache = jnp.zeros((pages + 1, PAGE // 4, 4, 256), jnp.uint8)
    targets = jnp.arange(1, pages + 1, dtype=jnp.int32)
    out = key_prologue_pallas(
        jnp.asarray(x), jnp.asarray(wk), jnp.asarray(ws), jnp.asarray(nw), jnp.asarray(nb),
        jnp.asarray(cos), jnp.asarray(sin), cache, targets, eps=EPS,
        interpret=jax.default_backend() != "tpu",
    )
    rec = np.asarray(out)[1:].reshape(T, 256)
    scale = np.ldexp(np.float32(1), rec[:, D].astype(np.int32) - 127)
    return rec[:, :D].copy().view(FP8).astype(np.float32) * scale[:, None]


def _shortfalls(keys, exact, seed):
    rng = np.random.default_rng(1000 + seed)
    q = rng.normal(size=(QUERIES, HEADS, D)).astype(np.float32)
    w = rng.uniform(0.1, 1.0, (QUERIES, HEADS)).astype(np.float32)

    def scores(k):
        return np.einsum("qhk,qh->qk", np.maximum(np.einsum("qhd,kd->qhk", q, k), 0), w)

    s_k, s_e = scores(keys), scores(exact)
    out = []
    for t in range(QUERIES):
        picks = np.argpartition(-s_k[t], TOPK)[:TOPK]
        kth = np.sort(s_e[t])[::-1][TOPK - 1]
        out.append(max(0.0, 1.0 - float(s_e[t][picks].min()) / float(kth)))
    return out


def _summary(v):
    v = np.asarray(v)
    return {
        "max": float(v.max()),
        "p99": float(np.quantile(v, 0.99)),
        "mean": float(v.mean()),
        "frac_over_0.02": float(np.mean(v > 0.02)),
        "frac_nonzero": float(np.mean(v > 0)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--out", default="/tmp/selection_bias.json")
    a = p.parse_args()

    res = {"target": [], "kernel": [], "target_vs_kernel_key_max_abs": []}
    for seed in range(a.seeds):
        inp = _inputs(seed)
        exact = _quantized(_exact_keys(*inp))
        target = _quantized(_target_keys(*inp))
        kernel = _kernel_keys(*inp)
        res["target"] += _shortfalls(target, exact, seed)
        res["kernel"] += _shortfalls(kernel, exact, seed)
        res["target_vs_kernel_key_max_abs"].append(float(np.abs(target - kernel).max()))
    report = {
        "device": str(jax.devices()[0]),
        "queries": len(res["target"]),
        "topk": TOPK,
        "target": _summary(res["target"]),
        "kernel": _summary(res["kernel"]),
        "target_vs_kernel_key_max_abs": max(res["target_vs_kernel_key_max_abs"]),
    }
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
