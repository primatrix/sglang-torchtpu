# SPDX-License-Identifier: Apache-2.0
"""The copied source scorer must pick its TPU kernels when a torch op exports it.

torch_tpu ops export their JAX function with ``JAX_PLATFORMS=cpu``. The source's
dispatch read ``jax.default_backend()`` / ``jax.devices()[0].device_kind``, so
under export every Pallas scorer, radix v2 and the SparseCore slot map fell back
to XLA without an error (the M1-S decode store had the same failure).
``target_device.export_target`` names the real chip for one export. Guarded:

* Under ``export_target`` the source's TPU kernels are selected, with the
  source's shape and hardware limits intact (radix v2 only on v7x).
* Outside it a CPU-backend call keeps the source's XLA scoring.

The kernels are replaced by stand-ins that record their calls; numerics are in
the TPU bench. Runs on the CPU backend, so it needs neither a TPU nor torch_tpu.
"""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit", chips=1)

assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "cpu"

import unittest
from unittest import mock

import jax.numpy as jnp

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas.source_dsa.kernels import radix_topk
from sglang_tpu.attention.pallas.source_dsa.kernels.dsa import indexer, physical_mapping
from sglang_tpu.attention.pallas.source_dsa.target_device import export_target

_P, _D, _H = 64, 128, 32


def _batch(*, tokens, seqs, pages_per_seq, decode):
    q_len = tokens // seqs
    return (
        jnp.zeros((tokens, _H, _D), jnp.float32),
        jnp.zeros((tokens, _H), jnp.float32),
        jnp.zeros((1 + seqs * pages_per_seq, _P, _D), jnp.bfloat16),
        jnp.full((seqs,), pages_per_seq * _P, jnp.int32),
        jnp.arange(1, 1 + seqs * pages_per_seq, dtype=jnp.int32),
        jnp.arange(seqs + 1, dtype=jnp.int32) * q_len,
        jnp.arange(seqs + 1, dtype=jnp.int32) * pages_per_seq * _P,
        jnp.array([seqs] * 3 if decode else [0, 0, seqs], jnp.int32),
    )


def _kernels_called(device_kind, *, tokens, seqs, pages_per_seq, k, decode, physical):
    """Names of the source kernels one call reaches, with call counts."""
    max_kv = pages_per_seq * _P
    called = {}

    def stub(name, make):
        def run(*args, **kwargs):
            called[name] = called.get(name, 0) + 1
            return make(*args, **kwargs)

        return run

    scores = lambda q, *a, **kw: jnp.zeros((q.shape[0], max_kv), jnp.float32)
    stubs = {
        (indexer, "paged_decode_scores_pallas"): stub("decode_score", scores),
        (indexer, "paged_extend_score_block_pallas"): stub("extend_score", scores),
        (indexer, "paged_extend_score_uniform_packed_pallas"): stub("packed_score", scores),
        (radix_topk, "radix_topk_v1"): stub("radix_v1", lambda s, **kw: jnp.zeros(s.shape, jnp.int32)),
        (radix_topk, "radix_topk_v2"): stub("radix_v2", lambda s, **kw: jnp.zeros((s.shape[0], k), jnp.int32)),
        (physical_mapping, "map_sparsecore"): stub("map_sparsecore", lambda i, *a, **kw: jnp.zeros(i.shape, jnp.int32)),
    }
    patches = [mock.patch.object(m, n, f) for (m, n), f in stubs.items()]
    for p in patches:
        p.start()
    try:
        indexer.compute_scores_and_select_topk_indices.clear_cache()
        args = _batch(tokens=tokens, seqs=seqs, pages_per_seq=pages_per_seq, decode=decode)
        kwargs = dict(k=k, pages_per_seq=pages_per_seq, one_token_per_seq=decode,
                      topk_impl="radix", output_physical_slots=physical)
        if device_kind is None:
            indexer.compute_scores_and_select_topk_indices(*args, **kwargs)
        else:
            with export_target(device_kind):
                indexer.compute_scores_and_select_topk_indices(*args, **kwargs)
    finally:
        for p in patches:
            p.stop()
        indexer.compute_scores_and_select_topk_indices.clear_cache()
    return called


_DECODE_2 = dict(tokens=2, seqs=2, pages_per_seq=4, k=128, decode=True, physical=True)
_DECODE_24 = dict(tokens=24, seqs=24, pages_per_seq=40, k=2048, decode=True, physical=False)
_EXTEND = dict(tokens=64, seqs=1, pages_per_seq=40, k=2048, decode=False, physical=False)
_PACKED = dict(tokens=12, seqs=2, pages_per_seq=40, k=2048, decode=False, physical=False)


class TestExportDispatch(CustomTestCase):
    def test_export_reaches_the_tpu_scorers(self):
        for name, case, scorer in (
            ("decode", _DECODE_24, "decode_score"),
            ("extend", _EXTEND, "extend_score"),
            ("packed extend", _PACKED, "packed_score"),
        ):
            with self.subTest(name):
                self.assertIn(scorer, _kernels_called("TPU7x", **case))

    def test_export_maps_physical_slots_on_the_sparsecore(self):
        self.assertIn("map_sparsecore", _kernels_called("TPU7x", **_DECODE_2))

    def test_radix_v2_only_on_v7x(self):
        for case in (_DECODE_24, _EXTEND):
            with self.subTest(tokens=case["tokens"]):
                v7x = _kernels_called("TPU7x", **case)
                v6e = _kernels_called("TPU v6 lite", **case)
                self.assertIn("radix_v2", v7x)
                self.assertNotIn("radix_v2", v6e)
                self.assertIn("radix_v1", v6e)

    def test_plain_cpu_call_keeps_the_xla_scorer(self):
        for case in (_DECODE_2, _DECODE_24, _EXTEND, _PACKED):
            with self.subTest(tokens=case["tokens"], decode=case["decode"]):
                called = _kernels_called(None, **case)
                for kernel in ("decode_score", "extend_score", "packed_score", "radix_v2",
                               "map_sparsecore"):
                    self.assertNotIn(kernel, called)

    def test_a_non_tpu_target_is_rejected(self):
        with self.assertRaises(ValueError):
            with export_target("cpu"):
                pass


if __name__ == "__main__":
    unittest.main()
