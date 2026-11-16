# SPDX-License-Identifier: Apache-2.0
"""Torch op for the source-contract DSA KV write + exact sparse attention (S5-S6).

``sglang_tpu_pallas::dsa_fp8_attention`` runs
``dsa_fp8_attention.fp8_attention_step``: quantize this step's latent + k_rope
into the split FP8 cache (u8 latent ``[P, 128, 4, 128]`` and RoPE
``[P, 32, 4, 128]``, both donated and aliased), map the indexer's
sequence-local top-k to physical rows, and run the source's
``sparse_core_tensor_core_dsa`` (SparseCore gather + TensorCore FP8 attention).

The op exports with ``JAX_PLATFORMS=cpu``. The copied attention reads the
hardware only through ``pltpu.get_tpu_info()`` (the SparseCore mesh in
``gather.py``), which ``tpu_info_for_cpu_export`` already answers with the
attached chip; ``export_target`` is entered as well so every source path sees
one target. Not wired into the attention backend yet.
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


def build_dsa_fp8_attention_op():
    """Register the op once per process and return it."""
    global _OP
    if _OP is not None:
        return _OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.dsa_fp8_attention import fp8_attention_step
    from sglang_tpu.attention.pallas.source_dsa.target_device import export_target

    op_name = "sglang_tpu_pallas::dsa_fp8_attention"

    def _wrapped(
        latent: jax.Array,
        rope: jax.Array,
        q_latent: jax.Array,
        q_rope: jax.Array,
        kv_c: jax.Array,
        k_pe: jax.Array,
        topk: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        sm_scale: float,
        allow_fp8_q: bool,
        is_prefill: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return fp8_attention_step(
            latent, rope, q_latent, q_rope, kv_c, k_pe, topk, kv_lens, page_indices,
            cu_q_lens, sm_scale, allow_fp8_q=allow_fp8_q, is_prefill=is_prefill,
        )

    latent_argnum, rope_argnum = argnums(_wrapped, "latent", "rope")
    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=argnums(_wrapped, "sm_scale", "allow_fp8_q", "is_prefill"),
        donate_argnums=(latent_argnum, rope_argnum),
        input_output_aliases={latent_argnum: 1, rope_argnum: 2},
        warn_deprecated=False,
    )

    @torch.library.custom_op(op_name, mutates_args=("latent", "rope"), device_types=["tpu"])
    def custom_op(
        latent: torch.Tensor,
        rope: torch.Tensor,
        q_latent: torch.Tensor,
        q_rope: torch.Tensor,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        topk: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        sm_scale: float,
        allow_fp8_q: bool,
        is_prefill: bool,
    ) -> torch.Tensor:
        # The target is the chip torch_tpu reports, not JAX's CPU export backend.
        with tpu_info_for_cpu_export(), export_target(get_pallas_tpu_device_kind()):
            out, _latent, _rope = jax_kernel(
                latent, rope, q_latent, q_rope, kv_c, k_pe, topk, kv_lens, page_indices,
                cu_q_lens, sm_scale, allow_fp8_q, is_prefill,
            )
        return out

    @custom_op.register_fake
    def _fake(latent, rope, q_latent, q_rope, kv_c, k_pe, topk, kv_lens, page_indices,
              cu_q_lens, sm_scale, allow_fp8_q, is_prefill):
        return q_latent.new_empty(q_latent.shape)

    _OP = custom_op
    return _OP
