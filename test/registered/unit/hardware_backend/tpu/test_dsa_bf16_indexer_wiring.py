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

from types import SimpleNamespace

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_pallas_mla_sparse import (
    TpuPallasSparseMlaAttnBackend,
    source_fp8_attention_problems,
)
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


_GLM52 = dict(
    bf16_indexer=True,
    kpool=1,
    kv_cache_dtype=torch.float8_e4m3fn,
    spec_enabled=False,
    zero_rope=False,
    attn_tp_size=1,
)


class TestSourceFp8AttentionGate(CustomTestCase):
    """SGLANG_TPU_DSA_SOURCE_FP8_ATTENTION must refuse configs it was not validated on.

    The attention reads the indexer's physical slots and writes the source's
    split-FP8 rows (per-token scale), so a config outside that contract would
    attend with wrong rows or a cache another writer cannot read, not fail.
    """

    def test_glm52_config_is_accepted(self):
        self.assertEqual(source_fp8_attention_problems(**_GLM52), [])

    def test_each_unsupported_setting_is_reported(self):
        for override in (
            dict(bf16_indexer=False),
            dict(kpool=4),
            dict(kv_cache_dtype=None),
            dict(kv_cache_dtype=torch.float8_e5m2),
            dict(spec_enabled=True),
            dict(zero_rope=True),
            dict(attn_tp_size=2),
        ):
            with self.subTest(**{k: str(v) for k, v in override.items()}):
                problems = source_fp8_attention_problems(**{**_GLM52, **override})
                self.assertEqual(len(problems), 1, problems)

    def test_mixed_batch_fails_instead_of_attending(self):
        backend = TpuPallasSparseMlaAttnBackend.__new__(TpuPallasSparseMlaAttnBackend)
        backend._source_fp8_attention = True
        backend._zero_rope = False
        layer = SimpleNamespace(layer_id=0, tp_q_head_num=2, v_head_dim=512, qk_head_dim=576)
        batch = SimpleNamespace(forward_mode=ForwardMode.MIXED, spec_info=None)
        q = torch.zeros(4, 2, 512, dtype=torch.bfloat16)
        k = torch.zeros(4, 1, 512, dtype=torch.bfloat16)
        with self.assertRaises(NotImplementedError):
            backend._forward_mla(
                q, k, layer, batch,
                q_rope=torch.zeros(4, 2, 64, dtype=torch.bfloat16),
                k_rope=torch.zeros(4, 1, 64, dtype=torch.bfloat16),
                topk_indices=torch.zeros(4, 8, dtype=torch.int32),
            )


if __name__ == "__main__":
    unittest.main()
