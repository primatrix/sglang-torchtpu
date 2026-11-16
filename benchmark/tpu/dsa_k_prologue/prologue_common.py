"""Shared inputs and NumPy references for the standalone K-prologue comparison.

Three implementations of "hidden -> index keys in the cache", each under its
own contract:

* Source: sglang-jax-private ``fused_fp8_k_projection_scatter_pallas``
  (W8A16 dot, eps 1e-5, bf16 norm params, Hadamard, bf16 cache, page 64).
* Target: the torch_tpu chain sglang-torchtpu runs today (``apply_tpu_fp8_linear``
  W8A8, f32 LayerNorm eps 1e-6, TPU rope) + ``quantize_fp8_ue8m0`` + ``_store_keys``
  (FP8 UE8M0 records, page 128).
* Ported: ``store_keys_fused`` (same contract as Target, fused kernel).

Every implementation is scored against an f64 reference of its own contract, so
the numbers say how close each comes to what it is meant to compute; the
selection shortfall is the metric all three share. NumPy only (no JAX/torch), so
the torch process and the JAX process build bit-identical inputs.
"""

import ml_dtypes
import numpy as np

HIDDEN, D, ROPE = 6144, 128, 64
EPS_TARGET, EPS_SOURCE = 1e-6, 1e-5
BF16, FP8 = ml_dtypes.bfloat16, ml_dtypes.float8_e4m3fn
PREFIX, CHUNK = 49152, 2048


def bf16(a):
    return np.asarray(a, np.float32).astype(BF16).astype(np.float32)


def inputs(tokens, seed=0):
    """Same weights for every point; hidden and rope angles per token count."""
    w = np.random.default_rng(seed)
    wk = w.normal(0, 0.02, (HIDDEN, D)).astype(FP8)  # [K, N]
    ws = w.uniform(0.5, 1.5, D).astype(np.float32)
    nw = w.uniform(0.5, 1.5, D).astype(np.float32)
    nb = w.normal(0, 0.1, D).astype(np.float32)
    r = np.random.default_rng(seed + tokens)
    x = r.normal(0, 1, (tokens, HIDDEN)).astype(BF16)
    # Per position angles; token j of every request sits at PREFIX + j.
    ang = r.uniform(0, 2 * np.pi, (PREFIX + CHUNK, ROPE // 2))
    table = np.concatenate([np.cos(ang), np.sin(ang)], -1).astype(BF16)  # cos_sin_cache
    q = tokens // max(1, tokens // CHUNK)
    pos = PREFIX + np.tile(np.arange(q), tokens // q)
    return dict(x=x, wk=wk, ws=ws, nw=nw, nb=nb, table=table, pos=pos)


def layout(tokens, page):
    """Requests of CHUNK new tokens after a page-aligned PREFIX; page table 1, 2, ..."""
    seqs = max(1, tokens // CHUNK)
    q = tokens // seqs
    pps = (PREFIX + q + page - 1) // page
    page_indices = (1 + np.arange(seqs * pps)).astype(np.int32)
    slot = []  # (physical page, slot) of each new token
    for s in range(seqs):
        p = PREFIX + np.arange(q)
        slot.append(np.stack([page_indices[s * pps + p // page], p % page], 1))
    return dict(
        seqs=seqs, q=q, pps=pps, num_pages=seqs * pps + 1, page_indices=page_indices,
        kv_lens=np.full(seqs, PREFIX + q, np.int32),
        cu_q_lens=(np.arange(seqs + 1) * q).astype(np.int32),
        distribution=np.array([0, 0, seqs], np.int32),
        slots=np.concatenate(slot),
    )


def hadamard():
    h = np.array([[1.0]])
    while h.shape[0] < D:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(D)).astype(np.float32)


def _rope_interleaved(k, cos, sin, rnd):
    x1, x2 = k[:, 0:ROPE:2], k[:, 1:ROPE:2]
    k = k.copy()
    k[:, 0:ROPE:2] = rnd(rnd(x1 * cos) - rnd(x2 * sin))
    k[:, 1:ROPE:2] = rnd(rnd(x2 * cos) + rnd(x1 * sin))
    return k


def target_reference(i):
    """f64 dot, the target chain's bf16 rounding elsewhere (as test_dsa_key_prologue)."""
    x = i["x"].astype(np.float32)
    xs = np.maximum(np.abs(x).max(1, keepdims=True) * (np.float32(1) / np.float32(448)), 1e-12)
    xq = (x / xs).astype(FP8).astype(np.float64)
    acc = (xq @ i["wk"].astype(np.float64)).astype(np.float32)
    k = bf16(acc * xs * i["ws"][None])
    mu, var = k.mean(1, keepdims=True), k.var(1, keepdims=True)
    k = bf16((k - mu) / np.sqrt(var + EPS_TARGET) * i["nw"] + i["nb"])
    cs = i["table"][i["pos"]].astype(np.float32)
    return _rope_interleaved(k, cs[:, : ROPE // 2], cs[:, ROPE // 2 :], bf16)


def source_reference(i):
    """Source contract in f64, bf16 only where its output is bf16; before Hadamard."""
    x = i["x"].astype(np.float64)
    k = bf16(x @ i["wk"].astype(np.float64) * i["ws"][None])  # BF16 projection output
    k = k.astype(np.float64)
    mu, var = k.mean(1, keepdims=True), k.var(1, keepdims=True)
    k = (k - mu) / np.sqrt(var + EPS_SOURCE) * bf16(i["nw"]) + bf16(i["nb"])
    cs = i["table"][i["pos"]].astype(np.float64)
    return _rope_interleaved(k, cs[:, : ROPE // 2], cs[:, ROPE // 2 :], lambda a: a)


def records(k):
    """FP8 UE8M0 records of keys: u8[T, 256] (quantize_fp8_ue8m0 + record layout)."""
    m, e = np.frexp(np.abs(k).max(1) / 448.0)
    e = np.clip(np.where(m == 0.5, e - 1, e), -126, 127)
    scale = np.ldexp(np.float32(1), e).astype(np.float32)
    rec = np.zeros((k.shape[0], 256), np.uint8)
    rec[:, :D] = np.clip(k / scale[:, None], -448, 448).astype(FP8).view(np.uint8)
    rec[:, D] = (e + 127).astype(np.uint8)
    return rec


def dequant(rec):
    scale = np.ldexp(np.float32(1), rec[:, D].astype(np.int32) - 127)
    return rec[:, :D].copy().view(FP8).astype(np.float32) * scale[:, None]


def shortfall(keys, exact, topk=128, heads=32, queries=512):
    """Per query 1 - (lowest exact score among the top-k over ``keys``) / (exact k-th best)."""
    r = np.random.default_rng(1)
    q = r.normal(size=(queries, heads, D)).astype(np.float32)
    w = r.uniform(0.1, 1.0, (queries, heads)).astype(np.float32)
    score = lambda k: np.einsum(
        "qhk,qh->qk", np.maximum(np.einsum("qhd,kd->qhk", q, k.astype(np.float32)), 0), w
    )
    s_k, s_e = score(keys), score(exact)
    picks = np.argpartition(-s_k, topk, axis=1)[:, :topk]
    lowest = np.take_along_axis(s_e, picks, axis=1).min(1)
    kth = -np.partition(-s_e, topk - 1, axis=1)[:, topk - 1]
    v = np.maximum(0.0, 1.0 - lowest / kth)
    return {"max": float(v.max()), "p99": float(np.quantile(v, 0.99)),
            "mean": float(v.mean()), "frac_over_2pct": float(np.mean(v > 0.02))}


def bf16_ulp(a, b):
    o = lambda x: (lambda i: np.where(i < 0, -(i & 0x7FFF), i))(
        np.asarray(x, np.float32).astype(BF16).view(np.int16).astype(np.int32))
    return np.abs(o(a) - o(b))


# Selection is scored on the first request's keys (CHUNK tokens): enough for 512
# queries x top-128, and the same size at every token count.
SELECT_TOKENS = CHUNK


def prefix_keys(num_tokens, seed):
    """bf16-valued N(0, 1) keys for the prefix, f32[num_tokens, 128].

    The benchmark's earlier prefix was random cache bytes. Those are not valid
    records: the scale byte names any power of two from 2^-127 to 2^128 and
    0x7F / 0xFF key bytes are FP8 NaN, so top-k scored garbage. The prefix is
    now real keys, written through the repo's own key store (``_store_keys``),
    so the byte layout is the one serving uses; this file does not restate it.
    """
    return bf16(np.random.default_rng(seed).normal(0, 1, (num_tokens, D)))


def prefix_bf16(num_pages, seed, page=64):
    """The Source's bf16 ``[P, page, 128]`` cache, filled with N(0, 1) keys."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (num_pages, page, D)).astype(BF16)
