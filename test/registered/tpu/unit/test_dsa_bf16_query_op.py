# SPDX-License-Identifier: Apache-2.0
"""The BF16 query-tail op must return the JAX function's result, eager and compiled.

``dsa_index_query_tail_bf16`` exports with ``JAX_PLATFORMS=cpu``; the prefill
kernel must still be the one that runs there (a backend check would silently
send every batch to the XLA chain, as happened to the key store's decode path).
Numerics are in test_dsa_bf16_query.py (raw JAX).
"""

import os
import unittest
from unittest import mock

# TorchTPU owns the device; JAX exports the custom call on the CPU.
os.environ["JAX_PLATFORMS"] = "cpu"

import torch
import torch_tpu  # noqa: F401

from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.dsa_bf16_query_op import build_dsa_index_query_tail_bf16_op

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

_T, _H, _D = 128, 32, 128


def _hadamard():
    h = torch.ones(1, 1)
    while h.shape[0] < _D:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h * _D**-0.5


def _inputs():
    gen = torch.Generator().manual_seed(0)
    angle = torch.rand(_T, 32, generator=gen) * 6.28
    return (
        torch.randn(_T, _H, _D, generator=gen).to(torch.bfloat16),
        angle.cos().to(torch.bfloat16),
        angle.sin().to(torch.bfloat16),
        _hadamard(),
    )


class TestBf16QueryTailOp(unittest.TestCase):
    def test_eager_and_compiled_agree_with_the_jax_function(self):
        import jax.numpy as jnp
        import ml_dtypes
        import numpy as np

        from sglang_tpu.attention.pallas.dsa_bf16_query import query_tail_xla

        op = build_dsa_index_query_tail_bf16_op()
        cpu = _inputs()
        dev = [t.to("tpu") for t in cpu]
        compiled = torch.compile(lambda *a: op(*a, True), backend="tpu", fullgraph=True, dynamic=False)
        eager = op(*dev, True).cpu()
        comp = compiled(*dev).cpu()
        self.assertEqual(eager.dtype, torch.float32)
        self.assertTrue(torch.equal(eager, comp), "eager and compiled differ")
        bf = lambda t: jnp.asarray(t.view(torch.int16).numpy().view(ml_dtypes.bfloat16))
        ref = np.asarray(query_tail_xla(bf(cpu[0]), bf(cpu[1]), bf(cpu[2]), jnp.asarray(cpu[3].numpy())))
        rel = np.abs(eager.numpy() - ref).max() / np.abs(ref).max()
        self.assertLess(rel, 0.01)


class TestPrefillDispatchUnderCpuExport(unittest.TestCase):
    def test_prefill_calls_the_source_kernel(self):
        import jax
        import jax.numpy as jnp

        from sglang_tpu.attention.pallas import dsa_bf16_query as dq

        self.assertEqual(jax.default_backend(), "cpu")
        q = jnp.zeros((_T, _H, _D), jnp.bfloat16)
        kernel = mock.Mock(return_value=q.astype(jnp.float32))
        with mock.patch.object(dq, "fused_q_rope_hadamard", kernel):
            dq.index_query_tail(q, jnp.ones((_T, 32)), jnp.zeros((_T, 32)), prefill=True)
        self.assertEqual(kernel.call_count, 1)
        self.assertEqual(kernel.call_args.kwargs["token_tile"], 64)


if __name__ == "__main__":
    unittest.main()
