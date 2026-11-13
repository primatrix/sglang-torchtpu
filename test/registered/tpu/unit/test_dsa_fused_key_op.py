# SPDX-License-Identifier: Apache-2.0
"""The fused-key indexer op: when it is used, and what the torch side must keep.

Numerics are in test_dsa_key_prologue.py. This file guards:

* The gate (``fused_key_prologue_params``). Only GLM-5.2's chain qualifies:
  switch on, interleaved rope, f32 LayerNorm, 128-wide keys, page 128, a
  channelwise FP8 ``wk`` without bias. Anything else must keep the torch chain.
* The op. The cache is donated and aliased, so a lost alias would leave the
  caller's tensor stale; eager and compiled must give the same rows and bytes.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

# TorchTPU owns the device; JAX exports the custom call on the CPU.
os.environ["JAX_PLATFORMS"] = "cpu"

import torch
import torch_tpu  # noqa: F401

from sglang.srt.layers.layernorm import LayerNorm, RMSNorm
from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.dsa_indexer_tpu import (
    build_dsa_index_topk_fused_key_op,
    fused_key_prologue_params,
)

register_tpu_ci(est_time=90, suite="stage-a-test-tpu-unit", chips=1)

_SWITCH = "SGLANG_TPU_DSA_FUSED_KEY_PROLOGUE"
_HIDDEN = 512
_D = 128
_ROPE = 64
_PAGE = 128


def _indexer(**change):
    """A GLM-5.2-shaped indexer; ``change`` overrides one property."""
    wk = SimpleNamespace(
        weight=torch.zeros(_HIDDEN, _D).to(torch.float8_e4m3fn),
        weight_scale=torch.ones(_D, 1),
        bias=None,
    )
    fields = dict(
        skip_rope=False,
        is_neox_style=False,
        head_dim=_D,
        rope_head_dim=_ROPE,
        hidden_size=_HIDDEN,
        use_dsa_indexer_fusion=False,
        k_norm=LayerNorm(_D, dtype=torch.float32),
        wk=wk,
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.zeros(16, _ROPE)),
    )
    fields.update(change)
    return SimpleNamespace(**fields)


class TestFusedKeyGate(unittest.TestCase):
    def test_glm52_chain_qualifies_only_with_the_switch_on(self):
        with mock.patch.dict(os.environ, {_SWITCH: "0"}):
            self.assertIsNone(fused_key_prologue_params(_indexer(), _PAGE))
        with mock.patch.dict(os.environ, {_SWITCH: "1"}):
            params = fused_key_prologue_params(_indexer(), _PAGE)
        self.assertIsNotNone(params)
        self.assertEqual(tuple(params["wk_scale"].shape), (_D,))
        self.assertEqual(params["eps"], 1e-6)

    def test_other_chains_keep_the_torch_path(self):
        fp8 = torch.float8_e4m3fn
        cases = {
            "NoPE (Kimi)": dict(skip_rope=True),
            "neox rope": dict(is_neox_style=True),
            "head_dim 64": dict(head_dim=64),
            "rope wider than half the key": dict(rope_head_dim=96),
            "wk folded into wk_weights_proj": dict(use_dsa_indexer_fusion=True),
            "RMSNorm": dict(k_norm=RMSNorm(_D)),
            "bf16 LayerNorm": dict(k_norm=LayerNorm(_D, dtype=torch.bfloat16)),
            "bf16 wk": dict(wk=SimpleNamespace(
                weight=torch.zeros(_HIDDEN, _D, dtype=torch.bfloat16),
                weight_scale=torch.ones(_D, 1), bias=None)),
            "block-scaled wk": dict(wk=SimpleNamespace(
                weight=torch.zeros(_HIDDEN, _D).to(fp8),
                weight_scale=torch.ones(1, 4, 4), bias=None)),
            "wk with bias": dict(wk=SimpleNamespace(
                weight=torch.zeros(_HIDDEN, _D).to(fp8),
                weight_scale=torch.ones(_D, 1), bias=torch.zeros(_D))),
            "no cos_sin_cache": dict(rotary_emb=SimpleNamespace()),
        }
        with mock.patch.dict(os.environ, {_SWITCH: "1"}):
            for name, change in cases.items():
                with self.subTest(name):
                    self.assertIsNone(fused_key_prologue_params(_indexer(**change), _PAGE))
            with self.subTest("page 64"):
                self.assertIsNone(fused_key_prologue_params(_indexer(), 64))


class TestFusedKeyOp(unittest.TestCase):
    def test_eager_and_compiled_update_the_original_cache(self):
        op = build_dsa_index_topk_fused_key_op()
        gen = torch.Generator().manual_seed(0)
        tokens, heads, topk, pages = 256, 8, 64, 6
        hidden = torch.randn(tokens, _HIDDEN, generator=gen).to(torch.bfloat16)
        wk = (torch.randn(_HIDDEN, _D, generator=gen) * 0.02).to(torch.float8_e4m3fn)
        angle = torch.rand(tokens, _ROPE // 2, generator=gen) * 6.28
        query = torch.randn(tokens, heads, _D, generator=gen).to(torch.bfloat16)
        # Two requests of 128 new tokens each -> pages 4 and 1; the rest untouched.
        page_indices = torch.tensor([4, 0, 0, 1, 0, 0], dtype=torch.int32)
        before = torch.randint(0, 256, (pages, _PAGE // 4, 4, 256), generator=gen,
                               dtype=torch.uint8)

        def args(cache):
            dev = lambda t: t.to("tpu")
            return (
                dev(query), dev(hidden), dev(wk), dev(torch.ones(_D)),
                dev(torch.ones(_D)), dev(torch.zeros(_D)),
                dev(angle.cos().to(torch.bfloat16)), dev(angle.sin().to(torch.bfloat16)),
                dev(torch.ones(tokens, heads)), cache,
                dev(torch.tensor([128, 128], dtype=torch.int32)), dev(page_indices),
                dev(torch.tensor([0, 128, 256], dtype=torch.int32)),
                dev(torch.tensor([0, 0, 2], dtype=torch.int32)),
                dev(torch.zeros(1, dtype=torch.int32)),
                1e-6, 1.0, topk, 1, "all", 0,
            )

        def run(*a):
            return op(*a)

        compiled = torch.compile(run, backend="tpu", fullgraph=True, dynamic=False)
        results = []
        for call in (run, compiled):
            cache = before.clone().to("tpu")
            rows = call(*args(cache)).cpu()
            results.append((rows, cache.cpu()))

        for rows, cache in results:
            self.assertEqual(tuple(rows.shape), (tokens, topk))
            # Written in place: the caller's tensor holds the new keys.
            for p in (4, 1):
                self.assertFalse(torch.equal(cache[p], before[p]), f"page {p} not written")
            for p in (0, 2, 3, 5):
                self.assertTrue(torch.equal(cache[p], before[p]), f"page {p} touched")
            # Each token sees its own request's keys only: 128 of them, < topk rows.
            self.assertTrue(bool((rows[:, : min(topk, 1)] >= 0).all()))
        (eager_rows, eager_cache), (comp_rows, comp_cache) = results
        self.assertTrue(torch.equal(eager_cache, comp_cache))
        self.assertTrue(torch.equal(eager_rows.sort(1).values, comp_rows.sort(1).values))


if __name__ == "__main__":
    unittest.main()
