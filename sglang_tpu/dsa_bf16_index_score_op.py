# SPDX-License-Identifier: Apache-2.0
"""Torch op for the source-contract indexer scoring + top-k (BF16 cache, page 64).

``sglang_tpu_pallas::dsa_index_score_topk_bf16`` runs the copied source
``compute_scores_and_select_topk_indices`` (``attention/pallas/source_dsa``)
with ``topk_impl="radix"``. The op exports with ``JAX_PLATFORMS=cpu``, so the
export runs under ``target_device.export_target`` with the device kind
torch_tpu reports for the attached chip; the source's Pallas / SparseCore
dispatch then sees the real target. Not wired into ``indexer_forward`` yet.
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


def build_dsa_index_score_topk_bf16_op():
    """Register the op once per process and return it."""
    global _OP
    if _OP is not None:
        return _OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.indexer import (
        compute_scores_and_select_topk_indices,
    )
    from sglang_tpu.attention.pallas.source_dsa.target_device import export_target

    op_name = "sglang_tpu_pallas::dsa_index_score_topk_bf16"

    def _wrapped(
        q_idx: jax.Array,
        idx_weights: jax.Array,
        cache: jax.Array,
        seq_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        cu_kv_lens: jax.Array,
        distribution: jax.Array,
        k: int,
        pages_per_seq: int,
        one_token_per_seq: bool,
        output_physical_slots: bool,
    ) -> jax.Array:
        return compute_scores_and_select_topk_indices(
            q_idx, idx_weights, cache, seq_lens, page_indices, cu_q_lens, cu_kv_lens,
            distribution, k=k, pages_per_seq=pages_per_seq,
            one_token_per_seq=one_token_per_seq, topk_impl="radix",
            output_physical_slots=output_physical_slots,
        )

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=argnums(
            _wrapped, "k", "pages_per_seq", "one_token_per_seq", "output_physical_slots"
        ),
        warn_deprecated=False,
    )

    @torch.library.custom_op(op_name, mutates_args=(), device_types=["tpu"])
    def custom_op(
        q_idx: torch.Tensor,
        idx_weights: torch.Tensor,
        cache: torch.Tensor,
        seq_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        cu_kv_lens: torch.Tensor,
        distribution: torch.Tensor,
        k: int,
        pages_per_seq: int,
        one_token_per_seq: bool = False,
        output_physical_slots: bool = False,
    ) -> torch.Tensor:
        # The target is the chip torch_tpu reports, not JAX's CPU export backend.
        with tpu_info_for_cpu_export(), export_target(get_pallas_tpu_device_kind()):
            return jax_kernel(
                q_idx, idx_weights, cache, seq_lens, page_indices, cu_q_lens, cu_kv_lens,
                distribution, k, pages_per_seq, one_token_per_seq, output_physical_slots,
            )

    @custom_op.register_fake
    def _fake(q_idx, idx_weights, cache, seq_lens, page_indices, cu_q_lens, cu_kv_lens,
              distribution, k, pages_per_seq, one_token_per_seq=False,
              output_physical_slots=False):
        return q_idx.new_empty((q_idx.shape[0], k), dtype=torch.int32)

    _OP = custom_op
    return _OP
