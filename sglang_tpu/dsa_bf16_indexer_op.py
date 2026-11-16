# SPDX-License-Identifier: Apache-2.0
"""Torch op for the source-contract DSA indexer (S1-S4) on the page-128 page table.

``sglang_tpu_pallas::dsa_bf16_indexer`` runs
``dsa_bf16_indexer.bf16_indexer_step``: store this step's index keys in
``cache`` (bf16 ``[P, 128, 128]``, donated and aliased) and return each query's
top-k sequence-local positions. The export runs under
``source_dsa.target_device.export_target`` with the chip torch_tpu reports, so
the source scorer and radix top-k dispatch for the real target. Not wired into
``indexer_forward`` yet.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here; custom_jax_kernel
# reads the ``jax.Array`` annotations of ``_wrapped``.

import torch

from sglang_tpu.pallas_export import (
    argnums,
    get_pallas_tpu_device_kind,
    tpu_info_for_cpu_export,
)

_OP = None


def build_dsa_bf16_indexer_op():
    """Register the op once per process and return it."""
    global _OP
    if _OP is not None:
        return _OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.dsa_bf16_indexer import bf16_indexer_step
    from sglang_tpu.attention.pallas.source_dsa.target_device import export_target

    op_name = "sglang_tpu_pallas::dsa_bf16_indexer"

    def _wrapped(
        cache: jax.Array,
        hidden: jax.Array,
        wk: jax.Array,
        wk_scale: jax.Array,
        norm_weight: jax.Array,
        norm_bias: jax.Array,
        query: jax.Array,
        rope_cos: jax.Array,
        rope_sin: jax.Array,
        head_weights: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        hadamard: jax.Array,
        topk: int,
        prefill: bool,
        decode: bool,
    ) -> tuple[jax.Array, jax.Array]:
        return bf16_indexer_step(
            cache, hidden, wk, wk_scale, norm_weight, norm_bias, query, rope_cos,
            rope_sin, head_weights, kv_lens, page_indices, cu_q_lens, distribution,
            hadamard, topk=topk, prefill=prefill, decode=decode,
        )

    (cache_argnum,) = argnums(_wrapped, "cache")
    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=argnums(_wrapped, "topk", "prefill", "decode"),
        donate_argnums=(cache_argnum,),
        input_output_aliases={cache_argnum: 1},
        warn_deprecated=False,
    )

    @torch.library.custom_op(op_name, mutates_args=("cache",), device_types=["tpu"])
    def custom_op(
        cache: torch.Tensor,
        hidden: torch.Tensor,
        wk: torch.Tensor,
        wk_scale: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        query: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        head_weights: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        distribution: torch.Tensor,
        hadamard: torch.Tensor,
        topk: int,
        prefill: bool,
        decode: bool,
    ) -> torch.Tensor:
        # The target is the chip torch_tpu reports, not JAX's CPU export backend.
        with tpu_info_for_cpu_export(), export_target(get_pallas_tpu_device_kind()):
            topk_indices, _cache = jax_kernel(
                cache, hidden, wk, wk_scale, norm_weight, norm_bias, query, rope_cos,
                rope_sin, head_weights, kv_lens, page_indices, cu_q_lens, distribution,
                hadamard, topk, prefill, decode,
            )
        return topk_indices

    @custom_op.register_fake
    def _fake(cache, hidden, wk, wk_scale, norm_weight, norm_bias, query, rope_cos,
              rope_sin, head_weights, kv_lens, page_indices, cu_q_lens, distribution,
              hadamard, topk, prefill, decode):
        return query.new_empty((query.shape[0], topk), dtype=torch.int32)

    _OP = custom_op
    return _OP
