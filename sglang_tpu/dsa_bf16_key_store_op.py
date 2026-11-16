# SPDX-License-Identifier: Apache-2.0
"""Torch op for the source-contract index-key prologue (BF16 cache, page 64).

``sglang_tpu_pallas::dsa_store_index_keys_bf16`` runs
``dsa_bf16_key_store.store_index_keys_bf16`` and updates ``cache`` in place
(donated and aliased). Not wired into ``indexer_forward``: the BF16 scoring it
pairs with is not migrated yet, so this op is validated on its own.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here; custom_jax_kernel
# reads the ``jax.Array`` annotations of ``_wrapped``.

import torch

from sglang_tpu.pallas_export import argnums, tpu_info_for_cpu_export

_OP = None


def build_dsa_store_index_keys_bf16_op():
    """Register the op once per process and return it."""
    global _OP
    if _OP is not None:
        return _OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.dsa_bf16_key_store import store_index_keys_bf16

    op_name = "sglang_tpu_pallas::dsa_store_index_keys_bf16"

    def _wrapped(
        cache: jax.Array,
        hidden: jax.Array,
        wk: jax.Array,
        wk_scale: jax.Array,
        norm_weight: jax.Array,
        norm_bias: jax.Array,
        rope_cos: jax.Array,
        rope_sin: jax.Array,
        hadamard: jax.Array,
        seq_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        cu_kv_lens: jax.Array,
    ) -> jax.Array:
        return store_index_keys_bf16(
            cache, hidden, wk, wk_scale, norm_weight, norm_bias, rope_cos, rope_sin,
            hadamard, seq_lens, page_indices, cu_q_lens, cu_kv_lens,
        )

    (cache_argnum,) = argnums(_wrapped, "cache")
    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        donate_argnums=(cache_argnum,),
        input_output_aliases={cache_argnum: 0},
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
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        hadamard: torch.Tensor,
        seq_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        cu_kv_lens: torch.Tensor,
    ) -> None:
        # JAX exports the custom call on the CPU and needs the TPU info to lower Mosaic.
        with tpu_info_for_cpu_export():
            jax_kernel(
                cache, hidden, wk, wk_scale, norm_weight, norm_bias, rope_cos, rope_sin,
                hadamard, seq_lens, page_indices, cu_q_lens, cu_kv_lens,
            )

    custom_op.register_fake(lambda *args: None)
    _OP = custom_op
    return _OP
