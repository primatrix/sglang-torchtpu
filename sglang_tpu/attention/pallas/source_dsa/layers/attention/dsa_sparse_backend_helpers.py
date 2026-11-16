# SPDX-License-Identifier: Apache-2.0
# Verbatim copies of three module-level functions of primatrix-glm/sglang-jax-private
# (Apache-2.0) at c948bdda95208e68a474c8088937347d08fe8937: python/sgl_jax/srt/layers/attention/dsa_sparse_backend.py
# (the module itself imports code not migrated here). Only the imports below are
# added; each function body is byte-identical (MANIFEST.json "extracts").
"""Source DSA sparse backend helpers: split-FP8 KV write, logical -> physical slots."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.fp8_kv_cache import (
    SplitFP8KVCache,
    quantize_fp8_kv_cache,
    quantize_split_fp8_kv_cache,
)
from sglang_tpu.attention.pallas.source_dsa.layers.attention.dsa_cache_ops import scatter_paged_cache


def _scatter_fused_kv_paged(
    cache4d: jax.Array | SplitFP8KVCache,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    *,
    kv_lora_rank: int,
    fp8_dtype=jnp.float8_e4m3fn,
) -> jax.Array | SplitFP8KVCache:
    """Quantize once, then scatter each physical buffer into the same pages."""
    if isinstance(cache4d, SplitFP8KVCache):
        packed = quantize_split_fp8_kv_cache(
            new_kv_c, new_k_pe, fp8_dtype=fp8_dtype or jnp.float8_e4m3fn
        )
        return SplitFP8KVCache(
            scatter_paged_cache(
                cache4d.latent, packed.latent, seq_lens, page_indices, cu_q_lens, cu_kv_lens
            ),
            scatter_paged_cache(
                cache4d.rope,
                packed.rope,
                seq_lens,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                tokens_per_group=4,
            ),
        )
    if cache4d.dtype == jnp.uint8:
        packed = quantize_fp8_kv_cache(
            new_kv_c,
            new_k_pe,
            fp8_dtype=fp8_dtype or jnp.float8_e4m3fn,
        )
        if packed.shape[1:] != cache4d.shape[2:]:
            raise ValueError(
                "FP8 KV payload shape does not match the cache: "
                f"payload={packed.shape[1:]}, cache={cache4d.shape[2:]}"
            )
        return scatter_paged_cache(
            cache4d,
            packed,
            seq_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
        )

    packing, packed_dim = cache4d.shape[-2:]
    packed = jnp.zeros((new_kv_c.shape[0], packing, packed_dim), dtype=cache4d.dtype)

    kv_positions = jnp.arange(kv_lora_rank, dtype=jnp.int32)
    packed = packed.at[:, kv_positions // packed_dim, kv_positions % packed_dim].set(
        new_kv_c.astype(cache4d.dtype)
    )

    rope_positions = kv_lora_rank + jnp.arange(new_k_pe.shape[-1], dtype=jnp.int32)
    packed = packed.at[:, rope_positions // packed_dim, rope_positions % packed_dim].set(
        new_k_pe.astype(cache4d.dtype)
    )
    return scatter_paged_cache(
        cache4d,
        packed,
        seq_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
    )


def _logical_topk_to_physical_slots(
    topk: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    page_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Map per-sequence logical token positions to flattened cache rows.

    ``page_indices`` is ragged-packed. Sequence ``i`` starts at
    ``cu_kv_lens[i] // page_size``; using a fixed ``i * pages_per_seq`` stride
    would silently cross sequence boundaries for unequal context lengths.
    """
    num_tokens, topk_size = topk.shape
    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
    seq_ids = jnp.searchsorted(cu_q_lens[1:], token_ids, side="right")
    seq_ids = jnp.clip(seq_ids, 0, seq_lens.shape[0] - 1)

    logical = jnp.maximum(topk, 0)
    page_ptr = cu_kv_lens[seq_ids, None] // page_size + logical // page_size
    ptr_in_bounds = (page_ptr >= 0) & (page_ptr < page_indices.shape[0])
    safe_ptr = jnp.clip(page_ptr, 0, page_indices.shape[0] - 1)
    physical_pages = page_indices[safe_ptr]
    query_valid = token_ids < cu_q_lens[-1]
    valid = (
        query_valid[:, None]
        & (topk >= 0)
        & (logical < seq_lens[seq_ids, None])
        & ptr_in_bounds
        & (physical_pages >= 0)
    )
    physical_slots = physical_pages * page_size + logical % page_size
    physical_slots = jnp.where(valid, physical_slots, jnp.int32(0))
    selected_counts = jnp.sum(valid, axis=1, dtype=jnp.int32)

    # The indexer compacts invalid candidates after selection, so valid entries
    # are a prefix as required by selected_counts. Their score order is not
    # relevant to sparse attention.
    del topk_size
    return physical_slots.astype(jnp.int32), selected_counts


def _pad_first_axis(x: jax.Array, multiple: int) -> jax.Array:
    pad = (-x.shape[0]) % multiple
    return jnp.pad(x, ((0, pad), *((0, 0),) * (x.ndim - 1)))
