# SPDX-License-Identifier: Apache-2.0
"""The fused index-key prologue must store what the unfused target chain stores.

``key_prologue_pallas`` replaces, for page-aligned prefill, the torch ``wk`` /
``k_norm`` / rope in ``indexer_forward`` plus ``quantize_fp8_ue8m0`` and
``_store_keys`` in ``dsa_indexer_step``. The reference here is NumPy that
follows those torch and JAX ops step by step, with its own page/slot addressing,
so it shares no code with the kernel. The tests guard what can go wrong silently:

* Addressing. Each token page goes to the page its request's page table names;
  a request's earlier pages, other pages and batch padding are not touched.
* The record. 128 FP8 key bytes, a UE8M0 scale byte, zeros: the layout the
  top-k kernel and the K3 megakernel read.
* The numbers. FP8 activation quantization, LayerNorm, interleaved rope and the
  UE8M0 scale must be the target's. The FP8 dot cannot be: on TPU both XLA's
  (the target's) and Mosaic's lose precision, in different elements. Keys are
  compared with an exact-dot reference within one FP8 step plus that dot error
  carried through LayerNorm and rope. On the same inputs and queries, the
  kernel's top-k selection must be no worse than the target chain's (XLA, the
  form that matched torch_tpu bit for bit on v7x): max shortfall, p99 shortfall
  and the fraction of queries more than 2% short, each without extra margin.
  This does not replace a comparison with the real torch_tpu chain.
* The alignment guard. Batches the kernel cannot write whole pages for must be
  reported as unaligned so the caller keeps the existing path.

Sizes are GLM-5.2's: hidden 6144, 128-wide keys, rope on 64 dims, page 128.
On a host without a TPU the kernel runs in Pallas interpret mode.
"""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

# Standalone raw-JAX kernel test: select the physical backend before importing JAX.
assert "jax" not in sys.modules
os.environ.setdefault("JAX_PLATFORMS", "tpu")

import unittest

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas.dsa_key_prologue import (
    key_prologue_pallas,
    key_prologue_targets,
)

_HIDDEN = 6144
_D = 128
_ROPE = 64
_PAGE = 128
_EPS = 1e-6  # LayerNorm default; GLM-5.2 sets no index_k_norm_eps.
_BF16 = ml_dtypes.bfloat16
_FP8 = ml_dtypes.float8_e4m3fn
_INTERPRET = jax.default_backend() != "tpu"
# Measured on v7x with these inputs: XLA's FP8 dot (the target) is up to 3.45
# off the exact sum, Mosaic's up to 3.11. The margin to 4.0 is arbitrary.
_DOT_ERR = 4.0
# The 2% line of test_dsa_index_topk.py; here only the cut for counting how many
# queries fall more than this far short, compared between kernel and target.
_SCORE_SLACK = 0.02


def _bf16(a):
    return np.asarray(a, np.float32).astype(_BF16).astype(np.float32)


def _reference_keys(x, wk, wk_scale, nw, nb, cos, sin):
    """f32 values of the bf16 key the target chain hands to quantize_fp8_ue8m0."""
    x = x.astype(np.float32)
    # fp8_linear.apply_tpu_fp8_linear; torch_tpu computes amax / 448 as
    # amax * (1 / 448) (measured on v7x), so the reference does too.
    amax = np.abs(x).max(1, keepdims=True)
    x_scale = np.maximum(amax * (np.float32(1) / np.float32(448.0)), np.float32(1e-12)).astype(np.float32)
    x_q = (x / x_scale).astype(_FP8).astype(np.float32)
    acc = (x_q.astype(np.float64) @ wk.astype(np.float32).astype(np.float64)).astype(np.float32)
    k = _bf16(acc * x_scale * wk_scale[None])
    # LayerNorm(dtype=float32), output cast back to bf16
    mu = k.mean(1, keepdims=True)
    var = ((k - mu) ** 2).mean(1, keepdims=True)
    k = _bf16((k - mu) / np.sqrt(var + _EPS) * nw + nb)
    # rotary_embedding.utils.apply_rotary_emb, GPT-J style, in bf16
    x1, x2 = k[:, 0:_ROPE:2], k[:, 1:_ROPE:2]
    c, s = _bf16(cos), _bf16(sin)
    o1 = _bf16(_bf16(x1 * c) - _bf16(x2 * s))
    o2 = _bf16(_bf16(x2 * c) + _bf16(x1 * s))
    k = k.copy()
    k[:, 0:_ROPE:2], k[:, 1:_ROPE:2] = o1, o2
    return k


def _target_keys(x, wk, wk_scale, nw, nb, cos, sin):
    """The target chain's keys: XLA, one jit per stage, host round trip between.

    On v7x this form matched torch_tpu's projection, LayerNorm and rope bit for
    bit; reduce_precision keeps the bf16 rounding XLA would otherwise fold away.
    """
    from jax import lax

    f32 = jnp.float32
    rt = lambda a: lax.reduce_precision(a.astype(f32), exponent_bits=8, mantissa_bits=7)

    @jax.jit
    def proj(x, wk, ws):
        xf = x.astype(f32)
        xs = jnp.maximum(jnp.max(jnp.abs(xf), axis=1, keepdims=True) / 448.0, 1e-12)
        xq = (xf / xs).astype(jnp.float8_e4m3fn)
        acc = lax.dot_general(xq, wk, (((1,), (0,)), ((), ())), preferred_element_type=f32)
        return rt(acc * xs * ws[None])

    @jax.jit
    def norm(k, nw, nb):
        mean = jnp.mean(k, axis=1, keepdims=True)
        var = jnp.mean(jnp.square(k - mean), axis=1, keepdims=True)
        return rt((k - mean) * lax.rsqrt(var + _EPS) * nw + nb)

    @jax.jit
    def rope(k, cos, sin):
        c, s = cos.astype(f32), sin.astype(f32)
        x1, x2 = k[:, 0:_ROPE:2], k[:, 1:_ROPE:2]
        o1 = rt(rt(x1 * c) - rt(x2 * s))
        o2 = rt(rt(x2 * c) + rt(x1 * s))
        rot = jnp.stack([o1, o2], axis=-1).reshape(k.shape[0], _ROPE)
        return jnp.concatenate([rot, k[:, _ROPE:]], axis=1)

    p = np.asarray(proj(jnp.asarray(x), jnp.asarray(wk), jnp.asarray(wk_scale)))
    n = np.asarray(norm(jnp.asarray(p), jnp.asarray(nw), jnp.asarray(nb)))
    return np.asarray(rope(jnp.asarray(n), jnp.asarray(cos), jnp.asarray(sin)))


def _selection_shortfall(keys, exact_k, topk=128, heads=32, queries=512):
    """Per query: 1 - (lowest exact score among the top-k picks over ``keys``)
    / (exact k-th best score). 0 means the picks are a true top-k.

    Scores are the indexer's: sum over heads of relu(q . k) * head weight. The
    queries are fixed by seed, so the kernel and the target see the same ones.
    """
    rng = np.random.default_rng(1)
    q = rng.normal(size=(queries, heads, _D)).astype(np.float32)
    w = rng.uniform(0.1, 1.0, (queries, heads)).astype(np.float32)

    def scores(k):
        return np.einsum("qhk,qh->qk", np.maximum(np.einsum("qhd,kd->qhk", q, k), 0), w)

    s_k, s_e = scores(keys), scores(exact_k)
    picks = np.argpartition(-s_k, topk, axis=1)[:, :topk]
    lowest = np.take_along_axis(s_e, picks, axis=1).min(1)
    kth = -np.partition(-s_e, topk - 1, axis=1)[:, topk - 1]
    return np.maximum(0.0, 1.0 - lowest / kth)


def _dot_error_bound(x, wk, wk_scale, nw):
    """Per-element bound on the key error a dot error of _DOT_ERR causes.

    First order: LayerNorm divides the projection error by the row's std and
    multiplies by the norm weight; rope adds the errors of a pair. Doubled
    (arbitrary) for the mean/variance shift and bf16 rounding flips.
    """
    x = x.astype(np.float32)
    x_scale = np.abs(x).max(1, keepdims=True) * (np.float32(1) / np.float32(448.0))
    x_q = (x / x_scale).astype(_FP8).astype(np.float64)
    k = (x_q @ wk.astype(np.float32).astype(np.float64)) * x_scale * wk_scale[None]
    e = _DOT_ERR * x_scale * wk_scale[None] * np.abs(nw)[None] / np.sqrt(k.var(1, keepdims=True) + _EPS)
    pair = e[:, 0:_ROPE:2] + e[:, 1:_ROPE:2]
    e = e.copy()
    e[:, 0:_ROPE:2], e[:, 1:_ROPE:2] = pair, pair
    return 2 * e


def _reference_records(k):
    """Index-key records of f32 keys: (u8[T, 256], scale f32[T])."""
    mantissa, exponent = np.frexp(np.abs(k).max(1) / 448.0)
    exponent = np.where(mantissa == 0.5, exponent - 1, exponent)
    exponent = np.clip(exponent, -126, 127)
    scale = np.ldexp(np.float32(1), exponent).astype(np.float32)
    values = np.clip(k / scale[:, None], -448, 448).astype(_FP8)
    rec = np.zeros((k.shape[0], 256), np.uint8)
    rec[:, :_D] = values.view(np.uint8)
    rec[:, _D] = (exponent + 127).astype(np.uint8)
    return rec, scale


def _dequant(records):
    scale = np.ldexp(np.float32(1), records[:, _D].astype(np.int32) - 127)
    return records[:, :_D].copy().view(_FP8).astype(np.float32) * scale[:, None]


class _Batch:
    """Two real requests and one padded request slot.

    Request 0: no prefix, 256 new tokens -> pages 5, 2.
    Request 1: a 128-token prefix in page 7, 256 new tokens -> pages 3, 8.
    The batch is padded to 640 tokens (one token page of padding).
    """

    num_tokens = 640
    num_pages = 10
    pages_per_seq = 4
    page_indices = np.array([5, 2, 0, 0, 7, 3, 8, 0, 0, 0, 0, 0], np.int32)
    kv_lens = np.array([256, 384, 0], np.int32)
    cu_q_lens = np.array([0, 256, 512, 512], np.int32)
    distribution = np.array([0, 0, 2], np.int32)
    # (request, cache page) of each of the batch's token pages; None = padding.
    expected_pages = [5, 2, 3, 8, None]


def _inputs(rng, num_tokens):
    x = rng.normal(0, 1, (num_tokens, _HIDDEN)).astype(_BF16)
    wk = rng.normal(0, 0.02, (_HIDDEN, _D)).astype(_FP8)
    wk_scale = rng.uniform(0.5, 1.5, _D).astype(np.float32)
    nw = rng.uniform(0.5, 1.5, _D).astype(np.float32)
    nb = rng.normal(0, 0.1, _D).astype(np.float32)
    angle = rng.uniform(0, 2 * np.pi, (num_tokens, _ROPE // 2))
    cos, sin = np.cos(angle).astype(_BF16), np.sin(angle).astype(_BF16)
    return x, wk, wk_scale, nw, nb, cos, sin


def _targets(b):
    return key_prologue_targets(
        b.num_tokens,
        jnp.asarray(b.kv_lens),
        jnp.asarray(b.page_indices),
        jnp.asarray(b.cu_q_lens),
        jnp.asarray(b.distribution),
        _PAGE,
    )


class TestDsaKeyPrologue(CustomTestCase):
    def test_stores_the_target_chain_records_in_the_right_pages(self):
        rng = np.random.default_rng(0)
        b = _Batch()
        x, wk, wk_scale, nw, nb, cos, sin = _inputs(rng, b.num_tokens)
        before = rng.integers(0, 256, (b.num_pages, _PAGE // 4, 4, 256), dtype=np.uint8)

        targets, aligned = _targets(b)
        self.assertTrue(bool(aligned))
        self.assertEqual(np.asarray(targets).tolist(), [p if p is not None else -1 for p in b.expected_pages])

        after = np.asarray(
            key_prologue_pallas(
                jnp.asarray(x), jnp.asarray(wk), jnp.asarray(wk_scale),
                jnp.asarray(nw), jnp.asarray(nb), jnp.asarray(cos), jnp.asarray(sin),
                jnp.asarray(before), targets, eps=_EPS, interpret=_INTERPRET,
            )
        )

        written = [p for p in b.expected_pages if p is not None]
        untouched = [p for p in range(b.num_pages) if p not in written]
        # Page 7 is request 1's prefix; page 0 is where padding would go.
        np.testing.assert_array_equal(after[untouched], before[untouched])

        ref_k = _reference_keys(x, wk, wk_scale, nw, nb, cos, sin)
        ref_rec, ref_scale = _reference_records(ref_k)
        got = np.stack([after[p].reshape(_PAGE, 256) for p in written])
        got = got.reshape(-1, 256)
        ref_rec, ref_scale, ref_k = ref_rec[: got.shape[0]], ref_scale[: got.shape[0]], ref_k[: got.shape[0]]

        np.testing.assert_array_equal(got[:, _D + 1 :], 0)
        scale_diff = got[:, _D].astype(np.int32) - ref_rec[:, _D]
        self.assertLessEqual(int(np.abs(scale_diff).max()), 1)
        self.assertGreaterEqual(float(np.mean(scale_diff == 0)), 0.99)
        # One FP8 step of the reference key (plus the subnormal floor), plus the
        # dot error carried to the key.
        got_k = _dequant(got)
        err = np.abs(got_k - ref_k)
        bound = 0.125 * np.abs(ref_k) + ref_scale[:, None] * 2.0**-8
        bound = bound + _dot_error_bound(x, wk, wk_scale, nw)[: got.shape[0]]
        self.assertTrue(np.all(err <= bound), f"max excess {float((err - bound).max())}")
        target_rec, _ = _reference_records(_target_keys(x, wk, wk_scale, nw, nb, cos, sin))
        self._assert_selection_no_worse_than_target(
            got_k=got_k,
            target_k=_dequant(target_rec[: got.shape[0]]),
            exact_k=_dequant(ref_rec),
        )

    def _assert_selection_no_worse_than_target(self, *, got_k, target_k, exact_k):
        """On the same queries, the kernel's top-k is no worse than the target's.

        The target's own FP8 dot already misses the exact top-k by up to ~5% on
        v7x, so a fixed slack cannot be the bar; the target's shortfall is.
        """
        shortfall = {
            "kernel": _selection_shortfall(got_k, exact_k),
            "target": _selection_shortfall(target_k, exact_k),
        }
        stats = {
            name: {
                "max": float(v.max()),
                "p99": float(np.quantile(v, 0.99)),
                "frac_over_2pct": float(np.mean(v > _SCORE_SLACK)),
            }
            for name, v in shortfall.items()
        }
        for stat in ("max", "p99", "frac_over_2pct"):
            self.assertLessEqual(
                stats["kernel"][stat], stats["target"][stat], f"{stat}: {stats}"
            )

    def test_unaligned_batch_falls_back_to_the_xla_chain(self):
        """store_keys_fused on a batch the kernel cannot take whole pages of.

        The XLA fallback must store each token in the slot its position names,
        leave every other slot alone, and select no worse than the target.
        Request 0: 200 new tokens -> page 5 slots 0..127, page 2 slots 0..71.
        Request 1: 128-token prefix, 256 new tokens -> pages 3 and 8.
        """
        from sglang_tpu.attention.pallas.dsa_index_topk import store_keys_fused

        rng = np.random.default_rng(2)
        b = _Batch()
        b.kv_lens = np.array([200, 384, 0], np.int32)
        b.cu_q_lens = np.array([0, 200, 456, 456], np.int32)
        _, aligned = _targets(b)
        self.assertFalse(bool(aligned))
        x, wk, wk_scale, nw, nb, cos, sin = _inputs(rng, b.num_tokens)
        before = rng.integers(0, 256, (b.num_pages, _PAGE // 4, 4, 256), dtype=np.uint8)

        after = np.asarray(
            store_keys_fused(
                jnp.asarray(before), jnp.asarray(x), jnp.asarray(wk),
                jnp.asarray(wk_scale), jnp.asarray(nw), jnp.asarray(nb),
                jnp.asarray(cos), jnp.asarray(sin), jnp.asarray(b.kv_lens),
                jnp.asarray(b.page_indices), jnp.asarray(b.cu_q_lens),
                jnp.asarray(b.distribution), eps=_EPS, interpret=_INTERPRET,
            )
        )
        rows = lambda c, p: c[p].reshape(_PAGE, 256)
        # (page, first slot, token range) of every stored run.
        runs = [(5, 0, range(0, 128)), (2, 0, range(128, 200)),
                (3, 0, range(200, 328)), (8, 0, range(328, 456))]
        written = {p for p, _, _ in runs}
        untouched = [p for p in range(b.num_pages) if p not in written]
        np.testing.assert_array_equal(after[untouched], before[untouched])
        # Page 2 past request 0's last token.
        np.testing.assert_array_equal(rows(after, 2)[72:], rows(before, 2)[72:])

        got = np.concatenate([rows(after, p)[s : s + len(t)] for p, s, t in runs])
        tokens = np.concatenate([np.asarray(t) for _, _, t in runs])
        np.testing.assert_array_equal(got[:, _D + 1 :], 0)
        ref_rec, _ = _reference_records(_reference_keys(x, wk, wk_scale, nw, nb, cos, sin))
        target_rec, _ = _reference_records(_target_keys(x, wk, wk_scale, nw, nb, cos, sin))
        self._assert_selection_no_worse_than_target(
            got_k=_dequant(got),
            target_k=_dequant(target_rec[tokens]),
            exact_k=_dequant(ref_rec[tokens]),
        )

    def test_unaligned_batches_are_reported(self):
        cases = {
            "prefix not on a page boundary": dict(kv_lens=[256, 320, 0], cu_q_lens=[0, 256, 512, 512]),
            "new tokens not whole pages": dict(kv_lens=[200, 384, 0], cu_q_lens=[0, 200, 456, 456]),
        }
        for name, change in cases.items():
            with self.subTest(name):
                b = _Batch()
                b.kv_lens = np.array(change["kv_lens"], np.int32)
                b.cu_q_lens = np.array(change["cu_q_lens"], np.int32)
                _, aligned = _targets(b)
                self.assertFalse(bool(aligned))

    def test_padded_request_slots_do_not_break_alignment(self):
        b = _Batch()
        # A padded slot with garbage lengths past distribution[2] is ignored.
        b.kv_lens = np.array([256, 384, 77], np.int32)
        _, aligned = _targets(b)
        self.assertTrue(bool(aligned))


if __name__ == "__main__":
    unittest.main()
