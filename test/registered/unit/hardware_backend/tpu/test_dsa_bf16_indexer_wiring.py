# SPDX-License-Identifier: Apache-2.0
"""SGLANG_TPU_DSA_BF16_INDEXER: the pool's bf16 index-key cache and the mode mapping.

With the switch on, every index-key buffer is the source indexer's bf16
``[P, 128, 128]`` (same 256 bytes per token as an FP8 record), a layout the
source chain does not cover fails at construction, and each forward mode maps
to the source's ``(prefill, decode)``. Off, the pool keeps the FP8 records.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.dsa_indexer_tpu import bf16_indexer_modes
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.kv_pool import TpuSplitMLATokenToKVPool

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SWITCH = tpu_envs.SGLANG_TPU_DSA_BF16_INDEXER


def _pool(*, page_size=128, kpool=1, head_dim=128, layers=(True, False, True)):
    return TpuSplitMLATokenToKVPool(
        size=4 * page_size,
        page_size=page_size,
        dtype=torch.float8_e4m3fn,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=len(layers),
        device="cpu",
        enable_memory_saver=False,
        index_head_dim=head_dim,
        index_key_layers=list(layers),
        index_kpool=kpool,
    )


class TestBf16IndexKeyPool(CustomTestCase):
    def test_switch_off_keeps_fp8_records(self):
        with SWITCH.override(False):
            pool = _pool()
        cache = pool.get_index_key_cache(0)
        self.assertEqual(cache.dtype, torch.uint8)
        self.assertEqual(tuple(cache.shape), (pool.num_pages, 32, 4, 256))

    def test_switch_on_allocates_bf16_rows(self):
        with SWITCH.override(True):
            pool = _pool()
        for layer in (0, 2):
            cache = pool.get_index_key_cache(layer)
            self.assertEqual(cache.dtype, torch.bfloat16)
            self.assertEqual(tuple(cache.shape), (pool.num_pages, 128, 128))
        # A layer that reuses the previous top-k still has no index-key cache.
        with self.assertRaises(ValueError):
            pool.get_index_key_cache(1)

    def test_bytes_per_token_match_the_fp8_record(self):
        with SWITCH.override(True):
            pool = _pool()
        bf16 = pool.get_index_key_cache(0)
        per_token = bf16.numel() * bf16.element_size() // (pool.num_pages * pool.page_size)
        self.assertEqual(per_token, TpuSplitMLATokenToKVPool.index_key_record_bytes(128))

    def test_unsupported_layouts_fail(self):
        for kw in (dict(kpool=4), dict(page_size=64), dict(head_dim=64)):
            with self.subTest(**kw), SWITCH.override(True), self.assertRaises(ValueError):
                _pool(**kw)


class TestBf16IndexerModes(CustomTestCase):
    def test_modes(self):
        expected = {
            ForwardMode.EXTEND: (True, False),
            ForwardMode.DECODE: (False, True),
            ForwardMode.MIXED: (False, False),
            ForwardMode.TARGET_VERIFY: (False, False),
            ForwardMode.DRAFT_EXTEND_V2: (False, False),
        }
        for mode, want in expected.items():
            with self.subTest(mode=mode.name):
                self.assertEqual(bf16_indexer_modes(mode), want)


if __name__ == "__main__":
    unittest.main()
