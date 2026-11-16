# SPDX-License-Identifier: Apache-2.0
# Verbatim copy of primatrix-glm/sglang-jax-private (Apache-2.0) at c948bdda95208e68a474c8088937347d08fe8937:
#   python/sgl_jax/srt/kernels/dsa/attention_inputs.py
# Only change: import prefix 'sgl_jax.srt.' -> 'sglang_tpu.attention.pallas.source_dsa.'. See source_dsa/MANIFEST.json.
"""Input contract shared by exact DSA scheduling implementations."""

import jax.numpy as jnp

from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.fp8_kv_cache import (
    FP8_KV_CACHE_NUM_LANES,
    FP8_KV_CACHE_SCALE_BYTES,
    SplitFP8KVCache,
    SplitFP8KVRows,
    cache_primary,
    split_fp8_kv_cache_layout,
    is_fp8_kv_cache_dtype,
)


def validate_attention_inputs(
    q_latent, q_rope, cache, physical_slots, selected_counts, *, fp8_dtype, bq_sparse, bq, b_topk
):
    if q_latent.dtype != jnp.bfloat16:
        raise TypeError("q_latent must have dtype bfloat16.")
    if q_rope.dtype != q_latent.dtype:
        raise TypeError("q_latent and q_rope must have the same dtype.")
    primary = cache_primary(cache)
    if isinstance(cache, SplitFP8KVCache):
        layout = split_fp8_kv_cache_layout(q_latent.shape[-1], q_rope.shape[-1])
        for part, shape in zip(cache, (layout.latent_shape, layout.rope_shape), strict=True):
            if part.dtype != jnp.uint8 or part.ndim != len(shape) + 1 or part.shape[1:] != shape:
                raise ValueError(
                    f"Split FP8 cache must use uint8 with shape [S, {', '.join(map(str, shape))}]."
                )
        if cache.latent.shape[0] != cache.rope.shape[0] * 4:
            raise ValueError("Split FP8 RoPE cache must contain one group per four latent slots.")
        fp8_dtype = fp8_dtype or jnp.float8_e4m3fn
        if not is_fp8_kv_cache_dtype(fp8_dtype):
            raise TypeError(f"Unsupported FP8 KV dtype: {fp8_dtype}.")
    else:
        if isinstance(cache, SplitFP8KVRows):
            raise ValueError(
                "Attention requires grouped RoPE storage, not per-token write payloads."
            )
        if cache.dtype not in (q_latent.dtype, jnp.uint8):
            raise TypeError("cache must have dtype bfloat16 or uint8.")
        if cache.dtype == jnp.uint8:
            fp8_dtype = fp8_dtype or jnp.float8_e4m3fn
            if not is_fp8_kv_cache_dtype(fp8_dtype):
                raise TypeError(f"Unsupported FP8 KV dtype: {fp8_dtype}.")
        elif fp8_dtype is not None:
            raise TypeError("fp8_dtype is only valid for a uint8 FP8 KV cache.")
        expected_pair_dim = FP8_KV_CACHE_NUM_LANES if cache.dtype == jnp.uint8 else 2
        if cache.ndim != 3 or cache.shape[1] != expected_pair_dim:
            raise ValueError(
                f"cache must have shape [S, {expected_pair_dim}, D], got {cache.shape}."
            )
        if cache.shape[-1] % 128:
            raise ValueError("cache pair dimension D must be divisible by 128.")
    if physical_slots.dtype != jnp.int32 or selected_counts.dtype != jnp.int32:
        raise TypeError("physical_slots and selected_counts must have dtype int32.")
    if bq_sparse <= 0 or bq <= 0 or b_topk <= 0:
        raise ValueError("bq_sparse, bq, and b_topk must be positive.")

    num_queries, _, latent_dim = q_latent.shape
    rope_dim = q_rope.shape[-1]
    cache_dim = (
        layout.row_bytes
        if isinstance(cache, SplitFP8KVCache)
        else primary.shape[1] * primary.shape[2]
    )
    topk = physical_slots.shape[1]
    required_cache_dim = latent_dim + rope_dim
    if fp8_dtype is not None:
        required_cache_dim += FP8_KV_CACHE_SCALE_BYTES
    if cache_dim < required_cache_dim:
        raise ValueError(
            f"cache dimension C={cache_dim} must be at least {required_cache_dim} "
            "to hold the latent and RoPE values and any FP8 scale."
        )
    if topk > 2048:
        raise ValueError(f"K must be at most 2048, got {topk}.")
    if num_queries % bq_sparse:
        raise ValueError(f"Q={num_queries} must be divisible by bq_sparse={bq_sparse}.")
    if bq_sparse % bq:
        raise ValueError(f"bq_sparse={bq_sparse} must be divisible by bq={bq}.")
    if topk % b_topk:
        raise ValueError(f"K={topk} must be divisible by b_topk={b_topk}.")

    return num_queries, latent_dim, rope_dim, topk
