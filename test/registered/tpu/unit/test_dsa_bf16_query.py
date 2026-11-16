# SPDX-License-Identifier: Apache-2.0
"""The source-contract indexer query tail must compute the source's rope + Hadamard.

``index_query_tail`` runs the source Pallas kernel on prefill batches of a
multiple of 64 tokens and the source's ``apply_rotary_emb`` + einsum otherwise.
The reference is NumPy in f64 with bf16 rounding only where the source rounds
(the rotated half before the f32 Hadamard). Guarded:

* Rope convention: interleaved (GPT-J) pairs on dims [:64]; dims [64:] untouched
  before the Hadamard. A neox-style or sign-flipped rotation fails the reference.
* Expanded coefficients: the kernel reads cos / sin repeated per pair and the
  sin signed (-s, s); a wrong expansion fails the kernel-vs-reference check.
* Dispatch: decode and token counts that are not a multiple of 64 take the
  XLA chain instead of failing; both paths agree.

Sizes are GLM-5.2's: 32 index heads, 128-wide queries, rope on 64 dims.
On a host without a TPU the kernel runs in Pallas interpret mode.
"""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit", chips=1)

# Standalone raw-JAX kernel test: select the physical backend before importing JAX.
assert "jax" not in sys.modules
os.environ.setdefault("JAX_PLATFORMS", "tpu")

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas import dsa_bf16_query as dq

_H, _D, _ROPE = 32, 128, 64
_BF16 = ml_dtypes.bfloat16
_INTERPRET = jax.default_backend() != "tpu"
# Max |error| over max |reference|; bf16 rotation output, f32 Hadamard. Measured
# 0.0016 in interpret mode; the margin to 0.01 is arbitrary.
_REL_TOL = 0.01


def _bf16(a):
    return np.asarray(a, np.float32).astype(_BF16).astype(np.float32)


def _inputs(tokens, seed=0):
    r = np.random.default_rng(seed)
    ang = r.uniform(0, 2 * np.pi, (tokens, _ROPE // 2))
    return dict(
        q=r.normal(0, 1, (tokens, _H, _D)).astype(_BF16),
        cos=np.cos(ang).astype(_BF16),
        sin=np.sin(ang).astype(_BF16),
    )


def _hadamard():
    h = np.array([[1.0]])
    while h.shape[0] < _D:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(_D)


def _reference(i):
    q = i["q"].astype(np.float64)
    c = i["cos"].astype(np.float64)[:, None, :]
    s = i["sin"].astype(np.float64)[:, None, :]
    x1, x2 = q[..., 0:_ROPE:2].copy(), q[..., 1:_ROPE:2].copy()
    q[..., 0:_ROPE:2] = _bf16(x1 * c - x2 * s)
    q[..., 1:_ROPE:2] = _bf16(x2 * c + x1 * s)
    return q @ _hadamard()


def _run(i, prefill):
    return np.asarray(
        dq.index_query_tail(
            jnp.asarray(i["q"]), jnp.asarray(i["cos"]), jnp.asarray(i["sin"]),
            prefill=prefill, interpret=_INTERPRET,
        )
    ).astype(np.float64)


class TestBf16QueryTail(CustomTestCase):
    def _check(self, tokens, prefill):
        i = _inputs(tokens)
        got, ref = _run(i, prefill), _reference(i)
        self.assertEqual(got.shape, (tokens, _H, _D))
        self.assertLess(np.abs(got - ref).max() / np.abs(ref).max(), _REL_TOL)
        return got

    def test_prefill_kernel_matches_the_reference(self):
        self._check(128, prefill=True)

    def test_decode_and_unaligned_prefill_take_the_xla_chain(self):
        kernel = mock.Mock(side_effect=AssertionError("kernel called"))
        with mock.patch.object(dq, "fused_q_rope_hadamard", kernel):
            self._check(8, prefill=False)
            self._check(40, prefill=True)
        kernel.assert_not_called()

    def test_kernel_and_xla_chain_agree(self):
        i = _inputs(64, seed=3)
        fused, xla = _run(i, prefill=True), _run(i, prefill=False)
        self.assertLess(np.abs(fused - xla).max() / np.abs(xla).max(), _REL_TOL)


if __name__ == "__main__":
    unittest.main()
