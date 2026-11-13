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
  UE8M0 scale must be the target's. The kernel's matmul sums in another order,
  so a key may land one bf16 step away and then round to the neighbouring FP8
  value; records are compared within one FP8 step, and almost all must match
  exactly.
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
        # Within one FP8 step of the reference bf16 key (plus the subnormal floor).
        err = np.abs(_dequant(got) - ref_k)
        bound = 0.125 * np.abs(ref_k) + ref_scale[:, None] * 2.0**-8
        self.assertTrue(np.all(err <= bound), f"max excess {float((err - bound).max())}")
        exact = np.mean(np.all(got == ref_rec, axis=1))
        self.assertGreaterEqual(float(exact), 0.95, f"exact records {exact:.3f}")

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
