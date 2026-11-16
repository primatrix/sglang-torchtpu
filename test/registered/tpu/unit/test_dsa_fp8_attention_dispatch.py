# SPDX-License-Identifier: Apache-2.0
"""The migrated split-FP8 attention step must export its TPU kernels from a CPU process.

torch_tpu ops export their JAX function with ``JAX_PLATFORMS=cpu``. The copied
source attention (``source_dsa/kernels/dsa``) has no backend branch: it reads
the hardware only through ``pltpu.get_tpu_info()`` (the SparseCore mesh in
``gather.py``). Under ``tpu_info_for_cpu_export`` that call answers for the
attached chip, so the exported module must hold the SparseCore gather and the
FP8 TensorCore attention for both the prefill plan and the decode plan.
Without the patch the export must fail (CPU has no SparseCore), not fall back
to something else silently.

The device kind is fixed to ``TPU7x`` (no torch_tpu needed). Runs on the CPU
backend; numerics are in the TPU bench.
"""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

assert "jax" not in sys.modules
os.environ["JAX_PLATFORMS"] = "cpu"

import unittest
from unittest import mock

import jax
import jax.numpy as jnp

from sglang.test.test_utils import CustomTestCase
from sglang_tpu import pallas_export
from sglang_tpu.attention.pallas import dsa_fp8_attention as port
from sglang_tpu.attention.pallas.source_dsa.target_device import export_target

_H, _LAT, _ROPE, _K = 64, 512, 64, 2048


def _args(*, tokens, seqs, pages_per_seq):
    pages = 1 + seqs * pages_per_seq
    q_len = tokens // seqs
    return (
        jnp.zeros((pages, 128, 4, 128), jnp.uint8),
        jnp.zeros((pages, 32, 4, 128), jnp.uint8),
        jnp.zeros((tokens, _H, _LAT), jnp.bfloat16),
        jnp.zeros((tokens, _H, _ROPE), jnp.bfloat16),
        jnp.zeros((tokens, _LAT), jnp.bfloat16),
        jnp.zeros((tokens, _ROPE), jnp.bfloat16),
        jnp.full((tokens, _K), -1, jnp.int32),
        jnp.full((seqs,), pages_per_seq * 128, jnp.int32),
        jnp.arange(1, pages, dtype=jnp.int32),
        jnp.arange(seqs + 1, dtype=jnp.int32) * q_len,
    )


def _export_text(*, is_prefill, **shape):
    fn = jax.jit(
        lambda *a: port.fp8_attention_step(
            *a, 192**-0.5, allow_fp8_q=True, is_prefill=is_prefill
        )
    )
    exported = jax.export.export(fn, platforms=["tpu"])(*_args(**shape))
    return exported.mlir_module()


_PREFILL = dict(is_prefill=True, tokens=256, seqs=1, pages_per_seq=24)
_DECODE = dict(is_prefill=False, tokens=4, seqs=4, pages_per_seq=24)


class TestFp8AttentionExport(CustomTestCase):
    def _patched(self):
        return mock.patch.object(
            pallas_export, "get_pallas_tpu_device_kind", return_value="TPU7x"
        )

    def test_export_holds_sparsecore_gather_and_fp8_attention(self):
        for name, case in (("prefill", _PREFILL), ("decode", _DECODE)):
            with self.subTest(name), self._patched():
                with pallas_export.tpu_info_for_cpu_export(), export_target("TPU7x"):
                    text = _export_text(**case)
                self.assertIn("dsa_sparse_core_gather", text)
                self.assertIn("dsa_tensor_core_attention_fp8", text)

    def test_export_without_the_patch_fails(self):
        with self.assertRaises(Exception):
            _export_text(**_DECODE)


if __name__ == "__main__":
    unittest.main()
