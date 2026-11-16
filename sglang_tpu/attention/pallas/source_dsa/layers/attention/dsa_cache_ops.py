# SPDX-License-Identifier: Apache-2.0
# Verbatim copy of primatrix-glm/sglang-jax-private (Apache-2.0) at c948bdda95208e68a474c8088937347d08fe8937:
#   python/sgl_jax/srt/layers/attention/dsa_cache_ops.py
# Only change: import prefix 'sgl_jax.srt.' -> 'sglang_tpu.attention.pallas.source_dsa.'. See source_dsa/MANIFEST.json.
"""Shared functional cache operations used by GLM-5.2 DSA paths."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def scatter_paged_cache(
    cache: jax.Array,
    new_tokens: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    *,
    tokens_per_group: int = 1,
) -> jax.Array:
    """Scatter tokens into a 3D index cache or a 4D packed KV cache.

    Supported layouts are ``[pages, page_size, dim]`` and
    ``[pages, page_size, packing, packed_dim]``. ``new_tokens`` must match the
    cache's trailing feature dimensions exactly; no broadcasting is allowed.
    With ``tokens_per_group > 1``, the layout is
    ``[pages, page_size // tokens_per_group, tokens_per_group, dim]``;
    each write updates only its token lane and preserves adjacent tokens.
    """

    if cache.ndim not in (3, 4):
        raise ValueError(f"cache must be rank 3 or 4, got shape {cache.shape}.")
    grouped = tokens_per_group != 1
    if tokens_per_group < 1 or (
        grouped and (cache.ndim != 4 or cache.shape[2] != tokens_per_group)
    ):
        raise ValueError("Grouped cache must have shape [pages, groups, tokens_per_group, dim].")
    feature_axes = 3 if grouped else 2
    if new_tokens.ndim != cache.ndim - feature_axes + 1:
        raise ValueError(
            f"new_tokens rank must be {cache.ndim - feature_axes + 1}, got "
            f"new_tokens.shape={new_tokens.shape} and cache.shape={cache.shape}."
        )
    if new_tokens.shape[1:] != cache.shape[feature_axes:]:
        raise ValueError(
            f"new_tokens trailing shape {new_tokens.shape[1:]} must match "
            f"cache slot shape {cache.shape[feature_axes:]}."
        )

    page_size = cache.shape[1] * tokens_per_group
    num_tokens = new_tokens.shape[0]
    num_seqs = seq_lens.shape[0]

    token_id = jnp.arange(num_tokens)
    seq_id = jnp.searchsorted(cu_q_lens[1:], token_id, side="right")
    seq_id = jnp.clip(seq_id, 0, num_seqs - 1)
    q_start = cu_q_lens[seq_id]
    q_end = cu_q_lens[seq_id + 1]
    kv_len = seq_lens[seq_id]
    abs_pos = jnp.maximum(kv_len - (q_end - q_start) + (token_id - q_start), 0)
    valid = (token_id >= q_start) & (token_id < q_end) & (kv_len > 0)

    page_local = abs_pos // page_size
    offset = abs_pos % page_size
    page = page_indices[cu_kv_lens[seq_id] // page_size + page_local]

    # Page 0 is the scratch page, NOT the top row. `PagedTokenToKVPoolAllocator`
    # hands out 1-based page ids `1 .. pages_per_rank` (allocator.py: `free_pages
    # = np.arange(1, pages_per_rank + 1)`), and `memory_pool._cache_shape` sizes
    # the page axis at `(size + page_size * dp_size) // page_size`, i.e. exactly
    # one spare row per DP rank. Sharded `P("data", ...)`, each shard therefore
    # holds rows `0 .. pages_per_rank` and the spare one is row 0 -- `shape[0] - 1`
    # is the highest page the allocator will legitimately hand out (host_kv_pool
    # `_require_device_pages` documents the same bound). Parking padded tokens
    # there overwrites slot 0 of a live sequence's KV once the free list has been
    # consumed far enough to reach that page. The read paths already use 0:
    # `_logical_topk_to_physical_slots` and the physical-top-k validation in
    # `dsa_sparse_backend` both fall back to slot 0.
    sentinel = 0
    safe_page = jnp.where(valid, page, sentinel)
    safe_offset = jnp.where(valid, offset, 0)
    if grouped:
        return cache.at[
            safe_page, safe_offset // tokens_per_group, safe_offset % tokens_per_group
        ].set(new_tokens.astype(cache.dtype))
    return cache.at[safe_page, safe_offset].set(new_tokens.astype(cache.dtype))
