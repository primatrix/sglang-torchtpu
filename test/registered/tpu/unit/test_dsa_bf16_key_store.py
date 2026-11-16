# SPDX-License-Identifier: Apache-2.0
"""The source-contract index-key prologue must store the source's keys where the page table says.

``store_index_keys_bf16`` (BF16 cache, page 64) runs the source Pallas kernel on
page-aligned extend batches and the source's XLA chain on the rest. The
reference is NumPy in f64 of the source contract (W8A16 dot, bf16 projection
output, LayerNorm eps 1e-5 with bf16 parameters, interleaved rope on dims
[:64], orthonormal Hadamard), with its own page/slot addressing. Guarded:

* Addressing: new tokens land in the pages their request's page table names;
  other pages, and the slots before a request's first new token, are untouched.
* Numbers: both paths within ``_REL_TOL`` of the f64 reference, and of each other.
* Dispatch: unaligned batches and token counts that are not a multiple of 64
  take the XLA chain instead of failing.

Sizes are GLM-5.2's: hidden 6144, 128-wide keys, rope on 64 dims.
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
from sglang_tpu.attention.pallas import dsa_bf16_key_store as ks

_HIDDEN, _D, _ROPE, _PAGE = 6144, 128, 64, 64
_BF16, _FP8 = ml_dtypes.bfloat16, ml_dtypes.float8_e4m3fn
_INTERPRET = jax.default_backend() != "tpu"
# Max |error| over max |reference|. The source kernel measured 0.0072-0.0100 on
# v7x (bf16 projection output and bf16 cache); the margin to 0.02 is arbitrary.
_REL_TOL = 0.02


def _bf16(a):
    return np.asarray(a, np.float32).astype(_BF16).astype(np.float32)


def _inputs(tokens, seed=0):
    r = np.random.default_rng(seed)
    return dict(
        x=r.normal(0, 1, (tokens, _HIDDEN)).astype(_BF16),
        wk=r.normal(0, 0.02, (_D, _HIDDEN)).astype(_FP8),  # [N, K], the source layout
        ws=r.uniform(0.5, 1.5, _D).astype(np.float32),
        nw=r.uniform(0.5, 1.5, _D).astype(_BF16),
        nb=r.normal(0, 0.1, _D).astype(_BF16),
        ang=r.uniform(0, 2 * np.pi, (tokens, _ROPE // 2)),
    )


def _hadamard():
    h = np.array([[1.0]])
    while h.shape[0] < _D:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(_D)


def _reference(i):
    """f64 keys after Hadamard; bf16 only where the source's output is bf16."""
    k = _bf16(i["x"].astype(np.float64) @ i["wk"].astype(np.float64).T * i["ws"]).astype(np.float64)
    k = (k - k.mean(1, keepdims=True)) / np.sqrt(k.var(1, keepdims=True) + 1e-5)
    k = k * i["nw"].astype(np.float64) + i["nb"].astype(np.float64)
    cos = _bf16(np.cos(i["ang"])).astype(np.float64)
    sin = _bf16(np.sin(i["ang"])).astype(np.float64)
    x1, x2 = k[:, 0:_ROPE:2].copy(), k[:, 1:_ROPE:2].copy()
    k[:, 0:_ROPE:2] = x1 * cos - x2 * sin
    k[:, 1:_ROPE:2] = x2 * cos + x1 * sin
    return k @ _hadamard()


def _args(i):
    return (
        jnp.asarray(i["x"]), jnp.asarray(i["wk"]), jnp.asarray(i["ws"]),
        jnp.asarray(i["nw"]), jnp.asarray(i["nb"]),
        jnp.asarray(np.cos(i["ang"]).astype(_BF16)), jnp.asarray(np.sin(i["ang"]).astype(_BF16)),
        ks.hadamard_128(),
    )


def _batch(prefixes, q_lens, pages_per_seq):
    """Metadata for requests with ``prefixes[s]`` cached and ``q_lens[s]`` new tokens.

    Request s owns pages ``1 + s * pages_per_seq ...`` in reverse order, so a
    wrong page-table lookup lands somewhere else. Returns the metadata and the
    (page, slot) of every new token.
    """
    tables, slots = [], []
    for s, (p, q) in enumerate(zip(prefixes, q_lens)):
        table = list(range(1 + (s + 1) * pages_per_seq - 1, s * pages_per_seq, -1))
        tables += table
        pos = p + np.arange(q)
        slots.append(np.stack([np.asarray(table)[pos // _PAGE], pos % _PAGE], 1))
    meta = (
        jnp.asarray(np.add(prefixes, q_lens), jnp.int32),
        jnp.asarray(tables, jnp.int32),
        jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32),
        jnp.asarray(np.arange(len(prefixes)) * pages_per_seq * _PAGE, jnp.int32),
    )
    return meta, np.concatenate(slots)


def _prefix_cache(num_pages, seed=1):
    return np.random.default_rng(seed).normal(0, 1, (num_pages, _PAGE, _D)).astype(_BF16)


class TestBf16KeyStore(CustomTestCase):
    def _check(self, prefixes, q_lens, pages_per_seq=8):
        tokens = int(sum(q_lens))
        i = _inputs(tokens)
        meta, slots = _batch(prefixes, q_lens, pages_per_seq)
        before = _prefix_cache(1 + len(prefixes) * pages_per_seq)
        after = np.asarray(
            ks.store_index_keys_bf16(jnp.asarray(before), *_args(i), *meta, interpret=_INTERPRET)
        ).astype(np.float32)
        ref = _reference(i)
        got = after[slots[:, 0], slots[:, 1]].astype(np.float64)
        rel = np.abs(got - ref).max() / np.abs(ref).max()
        self.assertLess(rel, _REL_TOL)
        written = np.zeros(after.shape[:2], bool)
        written[slots[:, 0], slots[:, 1]] = True
        np.testing.assert_array_equal(after[~written], before.astype(np.float32)[~written])
        return after, slots

    def test_aligned_batch_writes_the_named_pages(self):
        # Two requests, 2 and 3 whole new pages after 4- and 1-page prefixes.
        self._check(prefixes=[256, 64], q_lens=[128, 192])

    def test_unaligned_prefix_keeps_the_earlier_slots(self):
        # A 10-token tail on the last prefix page: the XLA chain writes from slot 10.
        self._check(prefixes=[266], q_lens=[128])

    def test_token_count_not_a_multiple_of_the_page(self):
        self._check(prefixes=[64, 0], q_lens=[37, 50])

    def test_kernel_and_xla_chain_agree(self):
        i = _inputs(256)
        meta, slots = _batch([128], [256], 8)
        before = jnp.asarray(_prefix_cache(9))
        fused = np.asarray(ks.store_index_keys_bf16(before, *_args(i), *meta, interpret=_INTERPRET))
        args = _args(i)
        xla = np.asarray(ks.scatter_paged_cache(before, ks.project_keys_xla(*args), *meta))
        a = fused[slots[:, 0], slots[:, 1]].astype(np.float32)
        b = xla[slots[:, 0], slots[:, 1]].astype(np.float32)
        self.assertLess(np.abs(a - b).max() / np.abs(b).max(), _REL_TOL)


if __name__ == "__main__":
    unittest.main()
