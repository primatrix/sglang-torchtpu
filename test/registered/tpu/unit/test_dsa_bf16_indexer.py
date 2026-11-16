# SPDX-License-Identifier: Apache-2.0
"""The page-128 adapter must run the source indexer exactly as page 64 would.

``bf16_indexer_step`` maps the framework's 128-token pages onto the source's
64-token pages (framework page ``p`` = source pages ``2p``, ``2p + 1``; the
cache reshaped in place) and chains the migrated key store, query tail and
scorer. The reference is the same three modules called directly at page 64 with
a page table and ``cu_kv_lens`` built here by hand, so a wrong page split,
offset or reshape changes the cache bytes or the top-k. Also guarded: the rows
are sequence-local positions inside each request (the format the target sparse
attention reads), and pages holding no new token are untouched.

On a TPU the scorer uses radix top-k; elsewhere the kernels run in Pallas
interpret mode and top-k is ``exact_lax`` (the SparseCore cannot be emulated).
"""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=90, suite="stage-a-test-tpu-unit", chips=1)

# Standalone raw-JAX kernel test: select the physical backend before importing JAX.
assert "jax" not in sys.modules
os.environ.setdefault("JAX_PLATFORMS", "tpu")

import unittest

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas import dsa_bf16_indexer as ix
from sglang_tpu.attention.pallas.dsa_bf16_key_store import hadamard_128, store_index_keys_bf16
from sglang_tpu.attention.pallas.dsa_bf16_query import index_query_tail
from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.indexer import (
    compute_scores_and_select_topk_indices,
)

_BF16, _FP8 = ml_dtypes.bfloat16, ml_dtypes.float8_e4m3fn
_INTERPRET = jax.default_backend() != "tpu"
_TOPK_IMPL = "exact_lax" if _INTERPRET else "radix"
_HIDDEN, _K = 512, 256


class TestBf16Indexer(CustomTestCase):
    def _run(self, q_lens, prefixes, *, decode, prefill, seed=0):
        r = np.random.default_rng(seed)
        T, S = sum(q_lens), len(q_lens)
        kv = np.add(prefixes, q_lens)
        ppr = int(-(-kv.max() // 128))
        # Request s owns framework pages in reverse order; page 0 is never real.
        pi = np.concatenate(
            [np.arange(1 + (s + 1) * ppr - 1, s * ppr, -1) for s in range(S)]
        ).astype(np.int32)
        P = 1 + S * ppr
        cache = jnp.asarray(r.normal(0, 1, (P, 128, 128)).astype(_BF16))
        hidden = jnp.asarray(r.normal(0, 1, (T, _HIDDEN)).astype(_BF16))
        wk = jnp.asarray(r.normal(0, 0.02, (128, _HIDDEN)).astype(_FP8))
        ws = jnp.asarray(r.uniform(0.5, 1.5, 128).astype(np.float32))
        nw = jnp.asarray(r.uniform(0.5, 1.5, 128).astype(_BF16))
        nb = jnp.asarray(r.normal(0, 0.1, 128).astype(_BF16))
        q = jnp.asarray(r.normal(0, 1, (T, 32, 128)).astype(_BF16))
        ang = r.uniform(0, 2 * np.pi, (T, 32))
        cos, sin = (jnp.asarray(f(ang).astype(_BF16)) for f in (np.cos, np.sin))
        w = jnp.asarray(r.uniform(0.1, 1.0, (T, 32)).astype(np.float32))
        cu_q = np.r_[0, np.cumsum(q_lens)].astype(np.int32)
        meta = (
            jnp.asarray(kv.astype(np.int32)), jnp.asarray(pi), jnp.asarray(cu_q),
            jnp.asarray(np.array([S] * 3 if decode else [0, 0, S], np.int32)),
        )
        H = hadamard_128()
        rows, out = ix.bf16_indexer_step(
            cache, hidden, wk, ws, nw, nb, q, cos, sin, w, *meta, H,
            topk=_K, prefill=prefill, decode=decode, topk_impl=_TOPK_IMPL,
            interpret=_INTERPRET,
        )

        # Reference: the same modules at page 64, table and offsets by hand.
        pi64 = jnp.asarray(np.stack([2 * pi, 2 * pi + 1], 1).reshape(-1).astype(np.int32))
        cukv = jnp.asarray((np.arange(S + 1) * ppr * 128).astype(np.int32))
        c64 = store_index_keys_bf16(
            cache.reshape(2 * P, 64, 128), hidden, wk, ws, nw, nb, cos, sin, H,
            meta[0], pi64, meta[2], cukv, one_token_per_seq=decode, interpret=_INTERPRET,
        )
        qi = index_query_tail(q, cos, sin, H, prefill=prefill, interpret=_INTERPRET)
        ref = compute_scores_and_select_topk_indices(
            qi, w, c64, meta[0], pi64, meta[2], cukv, meta[3], k=_K,
            pages_per_seq=2 * ppr, one_token_per_seq=decode, topk_impl=_TOPK_IMPL,
            output_physical_slots=False,
        )

        out_np, rows_np = np.asarray(out), np.asarray(rows)
        np.testing.assert_array_equal(
            out_np.view(np.int16), np.asarray(c64.reshape(cache.shape)).view(np.int16)
        )
        np.testing.assert_array_equal(np.sort(rows_np, 1), np.sort(np.asarray(ref), 1))
        seq = np.searchsorted(cu_q[1:], np.arange(T), side="right")
        for t in range(T):
            row = rows_np[t]
            self.assertTrue(((row == -1) | ((row >= 0) & (row < kv[seq[t]]))).all(), t)
        new_pages = {
            int(pi[s * ppr + pos // 128]) for s in range(S) for pos in range(prefixes[s], kv[s])
        }
        keep = [p for p in range(P) if p not in new_pages]
        np.testing.assert_array_equal(
            out_np[keep].view(np.int16), np.asarray(cache)[keep].view(np.int16)
        )

    def test_aligned_extend(self):
        # Whole 64-token source pages after 128-aligned prefixes: the fused key kernel.
        self._run([128, 256], [256, 128], decode=False, prefill=True)

    def test_unaligned_extend(self):
        # Prefixes mid-page and token counts off the 64 grid: the XLA key chain.
        self._run([100, 60], [200, 70], decode=False, prefill=True)

    def test_decode_across_the_half_page(self):
        # kv_len 301, 128 and 65: a new token in each half of a framework page.
        self._run([1, 1, 1], [300, 127, 64], decode=True, prefill=False)


if __name__ == "__main__":
    unittest.main()
