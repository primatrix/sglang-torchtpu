# SPDX-License-Identifier: Apache-2.0
"""Torch op for the source-contract indexer query tail (rope + Hadamard, f32 out).

``sglang_tpu_pallas::dsa_index_query_tail_bf16`` runs
``dsa_bf16_query.index_query_tail``. Not wired into ``indexer_forward``: the
BF16 scoring that reads its f32 output is not migrated yet, so the op is
validated on its own.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here; custom_jax_kernel
# reads the ``jax.Array`` annotations of ``_wrapped``.

import torch

from sglang_tpu.pallas_export import argnums, tpu_info_for_cpu_export

_OP = None


def build_dsa_index_query_tail_bf16_op():
    """Register the op once per process and return it."""
    global _OP
    if _OP is not None:
        return _OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.attention.pallas.dsa_bf16_query import index_query_tail

    op_name = "sglang_tpu_pallas::dsa_index_query_tail_bf16"

    def _wrapped(
        query: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        hadamard: jax.Array,
        prefill: bool,
    ) -> jax.Array:
        return index_query_tail(query, cos, sin, hadamard, prefill=prefill)

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=argnums(_wrapped, "prefill"),
        warn_deprecated=False,
    )

    @torch.library.custom_op(op_name, mutates_args=(), device_types=["tpu"])
    def custom_op(
        query: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        hadamard: torch.Tensor,
        prefill: bool = False,
    ) -> torch.Tensor:
        # JAX exports the custom call on the CPU and needs the TPU info to lower Mosaic.
        with tpu_info_for_cpu_export():
            return jax_kernel(query, cos, sin, hadamard, prefill)

    @custom_op.register_fake
    def _fake(query, cos, sin, hadamard, prefill=False):
        return query.new_empty(query.shape, dtype=torch.float32)

    _OP = custom_op
    return _OP
