# SPDX-License-Identifier: Apache-2.0
"""The BF16 index-key op must update the caller's cache, eager and compiled alike.

``dsa_store_index_keys_bf16`` donates and aliases ``cache``; a lost alias would
leave the caller's tensor stale while the op still "succeeds". Numerics are in
test_dsa_bf16_key_store.py (raw JAX); this file needs torch_tpu, so it is apart.
"""

import os
import unittest

# TorchTPU owns the device; JAX exports the custom call on the CPU.
os.environ["JAX_PLATFORMS"] = "cpu"

import torch
import torch_tpu  # noqa: F401

from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.dsa_bf16_key_store_op import build_dsa_store_index_keys_bf16_op

register_tpu_ci(est_time=90, suite="stage-a-test-tpu-unit", chips=1)

_HIDDEN, _D, _PAGE = 512, 128, 64


def _hadamard():
    h = torch.ones(1, 1)
    while h.shape[0] < _D:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h * _D**-0.5


class TestBf16KeyStoreOp(unittest.TestCase):
    def test_eager_and_compiled_update_the_original_cache(self):
        op = build_dsa_store_index_keys_bf16_op()
        gen = torch.Generator().manual_seed(0)
        tokens, pages = 128, 6
        # Two requests of 64 new tokens, no prefix -> pages 4 and 1; the rest untouched.
        before = torch.randn(pages, _PAGE, _D, generator=gen).to(torch.bfloat16)
        angle = torch.rand(tokens, 32, generator=gen) * 6.28
        dev = lambda t: t.to("tpu")
        inputs = (
            dev(torch.randn(tokens, _HIDDEN, generator=gen).to(torch.bfloat16)),
            dev((torch.randn(_D, _HIDDEN, generator=gen) * 0.02).to(torch.float8_e4m3fn)),
            dev(torch.rand(_D, generator=gen) + 0.5),
            dev(torch.ones(_D, dtype=torch.bfloat16)),
            dev(torch.zeros(_D, dtype=torch.bfloat16)),
            dev(angle.cos().to(torch.bfloat16)),
            dev(angle.sin().to(torch.bfloat16)),
            dev(_hadamard()),
            dev(torch.tensor([64, 64], dtype=torch.int32)),
            dev(torch.tensor([4, 0, 1, 0], dtype=torch.int32)),
            dev(torch.tensor([0, 64, 128], dtype=torch.int32)),
            dev(torch.tensor([0, 128], dtype=torch.int32)),  # cu_kv_lens: token offset of each page-table row
        )

        def run(cache, *a):
            op(cache, *a)

        compiled = torch.compile(run, backend="tpu", fullgraph=True, dynamic=False)
        results = []
        for call in (run, compiled):
            cache = before.clone().to("tpu")
            call(cache, *inputs)
            results.append(cache.cpu())

        for cache in results:
            for p in (0, 2, 3, 5):
                self.assertTrue(torch.equal(cache[p], before[p]), f"page {p} changed")
            for p in (1, 4):
                self.assertFalse(torch.equal(cache[p], before[p]), f"page {p} not written")
        self.assertTrue(torch.equal(results[0], results[1]), "eager and compiled differ")

    def test_decode_writes_only_each_requests_next_slot(self):
        op = build_dsa_store_index_keys_bf16_op()
        gen = torch.Generator().manual_seed(1)
        # Three decode requests: kv_len 10, 70 and 64 -> slots (page 2, 9), (page 5, 5), (page 3, 63).
        kv_lens = [10, 70, 64]
        tables = [[2, 0], [1, 5], [3, 0]]
        pages, tokens = 6, 3
        before = torch.randn(pages, _PAGE, _D, generator=gen).to(torch.bfloat16)
        angle = torch.rand(tokens, 32, generator=gen) * 6.28
        dev = lambda t: t.to("tpu")
        inputs = (
            dev(torch.randn(tokens, _HIDDEN, generator=gen).to(torch.bfloat16)),
            dev((torch.randn(_D, _HIDDEN, generator=gen) * 0.02).to(torch.float8_e4m3fn)),
            dev(torch.rand(_D, generator=gen) + 0.5),
            dev(torch.ones(_D, dtype=torch.bfloat16)),
            dev(torch.zeros(_D, dtype=torch.bfloat16)),
            dev(angle.cos().to(torch.bfloat16)),
            dev(angle.sin().to(torch.bfloat16)),
            dev(_hadamard()),
            dev(torch.tensor(kv_lens, dtype=torch.int32)),
            dev(torch.tensor(sum(tables, []), dtype=torch.int32)),
            dev(torch.tensor([0, 1, 2, 3], dtype=torch.int32)),
            dev(torch.tensor([0, 128, 256], dtype=torch.int32)),
        )

        def run(cache, *a):
            op(cache, *a, True)

        compiled = torch.compile(run, backend="tpu", fullgraph=True, dynamic=False)
        results = []
        for call in (run, compiled):
            cache = before.clone().to("tpu")
            call(cache, *inputs)
            results.append(cache.cpu())

        written = torch.zeros(pages, _PAGE, dtype=torch.bool)
        for (p, o) in ((2, 9), (5, 5), (3, 63)):
            written[p, o] = True
        for cache in results:
            self.assertTrue(torch.equal(cache[~written], before[~written]), "a slot outside the targets changed")
            self.assertFalse(torch.equal(cache[written], before[written]), "target slots not written")
        self.assertTrue(torch.equal(results[0], results[1]), "eager and compiled differ")


if __name__ == "__main__":
    unittest.main()
