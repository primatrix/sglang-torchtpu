# SPDX-License-Identifier: Apache-2.0
# Verbatim copy of primatrix-glm/sglang-jax-private (Apache-2.0) at c948bdda95208e68a474c8088937347d08fe8937:
#   python/sgl_jax/srt/kernels/dsa/fp8_kv_cache.py
# Only change: import prefix 'sgl_jax.srt.' -> 'sglang_tpu.attention.pallas.source_dsa.'. See source_dsa/MANIFEST.json.
"""FP8 storage helpers for the GLM-5.2 DSA KV cache.

Serving stores latent ``[P, page_size, 4, 128]`` and RoPE
``[P, page_size // 4, 4, 128]``. Both parts share the original FP32 row scale, stored
in RoPE bytes [64:68]; its remaining bytes are padding. SparseCore gathers
both buffers separately and TensorCore decodes their loaded VMEM tiles.

The uint8 storage preserves FP8 value bits and the mixed FP32 scale.
Legacy fused-row helpers remain available for numerical/performance comparison.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp
from jax import lax


FP8_KV_CACHE_NUM_LANES = 4
FP8_KV_CACHE_SCALE_BYTES = 4
_FP8_KV_CACHE_PAIR_ALIGNMENT = 128


class FP8KVCacheLayout(NamedTuple):
    """Physical byte layout for one DSA KV row."""

    kv_lora_rank: int
    qk_rope_head_dim: int
    quantized_dim: int
    scale_offset: int
    payload_bytes: int
    row_bytes: int
    pair_dim: int


def is_fp8_kv_cache_dtype(dtype) -> bool:
    """Whether ``dtype`` is one of the JAX FP8 dtypes supported for KV."""

    return jnp.dtype(dtype) in (
        jnp.dtype(jnp.float8_e4m3fn),
        jnp.dtype(jnp.float8_e5m2),
    )


def fp8_kv_cache_layout(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> FP8KVCacheLayout:
    """Return the byte layout used for a quantized DSA KV row.

    ``row_bytes`` is padded to four 128-byte SparseCore lanes, so the result
    can be represented by the exact-DSA FP8 cache shape
    ``[pages, page_size, 4, pair_dim]``.  For GLM-5.2 (512 latent + 64 RoPE)
    this is ``[..., 4, 256]``: 580 payload bytes and 444 bytes of padding.
    """

    if kv_lora_rank <= 0 or qk_rope_head_dim <= 0:
        raise ValueError("KV latent and RoPE dimensions must be positive.")
    quantized_dim = kv_lora_rank + qk_rope_head_dim
    scale_offset = quantized_dim
    payload_bytes = scale_offset + FP8_KV_CACHE_SCALE_BYTES
    row_alignment = FP8_KV_CACHE_NUM_LANES * _FP8_KV_CACHE_PAIR_ALIGNMENT
    row_bytes = (payload_bytes + row_alignment - 1) // row_alignment * row_alignment
    return FP8KVCacheLayout(
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        quantized_dim=quantized_dim,
        scale_offset=scale_offset,
        payload_bytes=payload_bytes,
        row_bytes=row_bytes,
        pair_dim=row_bytes // FP8_KV_CACHE_NUM_LANES,
    )


def _as_byte_view(value: jnp.ndarray, dtype, element_count: int) -> jnp.ndarray:
    """Bitcast a value array to a flat byte array along its final axis."""

    value = value.astype(dtype)
    byte_view = lax.bitcast_convert_type(value, jnp.uint8)
    return byte_view.reshape((*value.shape[:-1], element_count))


def _fp8_max(dtype) -> float:
    if not is_fp8_kv_cache_dtype(dtype):
        raise TypeError(f"Unsupported FP8 KV dtype: {dtype}.")
    return float(jnp.finfo(dtype).max)


def quantize_fp8_kv_cache(
    cache_k_nope: jnp.ndarray,
    cache_k_rope: jnp.ndarray,
    *,
    fp8_dtype=jnp.float8_e4m3fn,
) -> jnp.ndarray:
    """Quantize and pack one latent-plus-RoPE row into exact-DSA byte rows.

    Args:
      cache_k_nope: ``[..., kv_lora_rank]`` latent/nope vectors.
      cache_k_rope: ``[..., qk_rope_head_dim]`` RoPE vectors.
      fp8_dtype: FP8 format used for the complete latent+RoPE row.

    Returns:
      ``uint8`` rows with shape ``[..., 4, pair_dim]``. The FP32 scale at
      ``layout.scale_offset`` is shared by every FP8 value in that row.
    """

    if not is_fp8_kv_cache_dtype(fp8_dtype):
        raise TypeError(f"Unsupported FP8 KV dtype: {fp8_dtype}.")
    if cache_k_nope.ndim == 0 or cache_k_rope.ndim == 0:
        raise ValueError("FP8 KV inputs must have a final channel dimension.")
    if cache_k_nope.shape[:-1] != cache_k_rope.shape[:-1]:
        raise ValueError(
            "Latent and RoPE KV inputs must have matching leading dimensions, "
            f"got {cache_k_nope.shape} and {cache_k_rope.shape}."
        )

    layout = fp8_kv_cache_layout(cache_k_nope.shape[-1], cache_k_rope.shape[-1])
    values = jnp.concatenate(
        (cache_k_nope.astype(jnp.float32), cache_k_rope.astype(jnp.float32)),
        axis=-1,
    )
    max_abs = jnp.max(jnp.abs(values), axis=-1, keepdims=True)
    scales = (max_abs / _fp8_max(fp8_dtype)).astype(jnp.float32)
    inverse_scales = jnp.where(scales > 0, jnp.reciprocal(scales), 0.0)
    quantized = jnp.clip(
        values * inverse_scales,
        -_fp8_max(fp8_dtype),
        _fp8_max(fp8_dtype),
    ).astype(fp8_dtype)

    value_bytes = _as_byte_view(quantized, fp8_dtype, layout.quantized_dim)
    scale_bytes = _as_byte_view(scales, jnp.float32, FP8_KV_CACHE_SCALE_BYTES)
    payload = jnp.concatenate((value_bytes, scale_bytes), axis=-1)
    padding = layout.row_bytes - layout.payload_bytes
    if padding:
        payload = jnp.pad(
            payload,
            ((0, 0),) * (payload.ndim - 1) + ((0, padding),),
        )
    return payload.reshape(*payload.shape[:-1], FP8_KV_CACHE_NUM_LANES, layout.pair_dim)


def _flatten_cache_bytes(cache: jnp.ndarray) -> jnp.ndarray:
    if cache.ndim >= 2 and cache.shape[-2] in (2, FP8_KV_CACHE_NUM_LANES):
        return cache.reshape(*cache.shape[:-2], cache.shape[-2] * cache.shape[-1])
    return cache


def dequantize_fp8_kv_cache(
    cache: jnp.ndarray,
    *,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    fp8_dtype=jnp.float8_e4m3fn,
    output_dtype=jnp.bfloat16,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Decode byte rows into logical latent KV and RoPE tensors.

    This helper is shared by correctness tests and the TensorCore kernel's
    device-side decode path.
    """

    if isinstance(cache, (SplitFP8KVCache, SplitFP8KVRows)):
        return dequantize_split_fp8_kv_cache(
            cache,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            fp8_dtype=fp8_dtype,
            output_dtype=output_dtype,
        )
    if cache.dtype != jnp.uint8:
        raise TypeError(f"Quantized KV storage must be uint8, got {cache.dtype}.")
    if not is_fp8_kv_cache_dtype(fp8_dtype):
        raise TypeError(f"Unsupported FP8 KV dtype: {fp8_dtype}.")
    layout = fp8_kv_cache_layout(kv_lora_rank, qk_rope_head_dim)
    flat = _flatten_cache_bytes(cache)
    if flat.shape[-1] < layout.payload_bytes:
        raise ValueError(
            f"Quantized KV row has {flat.shape[-1]} bytes, needs at least {layout.payload_bytes}."
        )

    quantized = lax.bitcast_convert_type(flat[..., : layout.quantized_dim], fp8_dtype).astype(
        jnp.float32
    )
    scale_bytes = flat[..., layout.scale_offset : layout.payload_bytes].reshape(
        *flat.shape[:-1], 1, FP8_KV_CACHE_SCALE_BYTES
    )
    scales = lax.bitcast_convert_type(scale_bytes, jnp.float32)
    values = (quantized * scales).astype(output_dtype)
    nope = values[..., :kv_lora_rank]
    rope = values[..., kv_lora_rank : kv_lora_rank + qk_rope_head_dim]
    return nope, rope


def expand_fp8_kv_cache_for_attention(
    cache: jnp.ndarray, *, kv_lora_rank: int, qk_rope_head_dim: int, fp8_dtype=jnp.float8_e4m3fn
) -> jnp.ndarray:
    """Transient BF16 view of stored FP8 values for reuse across extend queries.

    This does not requantize or update the persistent cache. The result uses
    the existing two-lane BF16 SparseCore layout.
    """
    latent, rope = dequantize_fp8_kv_cache(
        cache, kv_lora_rank=kv_lora_rank, qk_rope_head_dim=qk_rope_head_dim, fp8_dtype=fp8_dtype
    )
    values = jnp.concatenate((latent, rope), axis=-1)
    row_size = (values.shape[-1] + 255) // 256 * 256
    values = jnp.pad(values, ((0, 0),) * (values.ndim - 1) + ((0, row_size - values.shape[-1]),))
    return values.reshape(*values.shape[:-1], 2, row_size // 2)


class SplitFP8KVCache(NamedTuple):
    """Two byte buffers containing FP8 latent values and RoPE plus row scale.

    GLM-5.2 uses latent ``[P, page_size, 4, 128]`` and RoPE
    ``[P, page_size // 4, 4, 128]``. Without page axes these are
    ``[S, 4, 128]`` and ``[S // 4, 4, 128]`` respectively.
    The shared FP32 scale occupies rope bytes [64:68]. Neither buffer stores
    uint32 words or needs last-axis padding at the TensorCore boundary.
    """

    latent: jnp.ndarray
    rope: jnp.ndarray


class SplitFP8KVRows(NamedTuple):
    """Transient per-token quantization or compact gather output, not storage.

    Latent has shape ``[..., 4, 128]`` and RoPE has shape ``[..., 128]``.
    Scatter writes each RoPE row into one lane of a persistent four-token group.
    """

    latent: jnp.ndarray
    rope: jnp.ndarray


class SplitFP8KVCacheLayout(NamedTuple):
    latent_shape: tuple[int, int]
    rope_shape: tuple[int, int]
    scale_offset: int
    row_bytes: int


class GatheredFP8KVGroups(NamedTuple):
    """Experimental gathered payload: one latent row and four RoPE rows per key.

    ``rope_lanes`` selects the source slot modulo four. This transient buffer
    is consumed inside TensorCore; it is not the persistent cache format.
    """

    latent: jnp.ndarray
    rope: jnp.ndarray
    rope_lanes: jnp.ndarray


def split_fp8_kv_cache_layout(kv_lora_rank: int, qk_rope_head_dim: int) -> SplitFP8KVCacheLayout:
    if kv_lora_rank <= 0 or qk_rope_head_dim <= 0:
        raise ValueError("KV latent and RoPE dimensions must be positive.")
    latent_bytes = (kv_lora_rank + 511) // 512 * 512
    rope_bytes = (qk_rope_head_dim + FP8_KV_CACHE_SCALE_BYTES + 127) // 128 * 128
    return SplitFP8KVCacheLayout(
        (4, latent_bytes // 4), (4, rope_bytes), qk_rope_head_dim, latent_bytes + rope_bytes
    )


def map_cache_parts(fn, cache):
    """Apply an array/Ref operation independently to each physical KV buffer."""
    if isinstance(cache, (SplitFP8KVCache, SplitFP8KVRows, GatheredFP8KVGroups)):
        return type(cache)(*(fn(part) for part in cache))
    return fn(cache)


def cache_parts(cache):
    return (
        tuple(cache)
        if isinstance(cache, (SplitFP8KVCache, SplitFP8KVRows, GatheredFP8KVGroups))
        else (cache,)
    )


def cache_primary(cache):
    return (
        cache.latent
        if isinstance(cache, (SplitFP8KVCache, SplitFP8KVRows, GatheredFP8KVGroups))
        else cache
    )


def cache_nbytes(cache):
    return sum(part.size * part.dtype.itemsize for part in cache_parts(cache))


def quantize_split_fp8_kv_cache(
    cache_k_nope: jnp.ndarray, cache_k_rope: jnp.ndarray, *, fp8_dtype=jnp.float8_e4m3fn
) -> SplitFP8KVRows:
    """Return per-token write payloads with a shared scale; scatter groups RoPE."""
    if cache_k_nope.ndim == 0 or cache_k_rope.ndim == 0:
        raise ValueError("FP8 KV inputs must have a final channel dimension.")
    if cache_k_nope.shape[:-1] != cache_k_rope.shape[:-1]:
        raise ValueError("Latent and RoPE KV inputs must have matching leading dimensions.")
    layout = split_fp8_kv_cache_layout(cache_k_nope.shape[-1], cache_k_rope.shape[-1])
    values = jnp.concatenate(
        (cache_k_nope.astype(jnp.float32), cache_k_rope.astype(jnp.float32)), axis=-1
    )
    scales = jnp.max(jnp.abs(values), axis=-1, keepdims=True) / _fp8_max(fp8_dtype)
    inverse_scales = jnp.where(scales > 0, jnp.reciprocal(scales), 0.0)
    values = jnp.clip(values * inverse_scales, -_fp8_max(fp8_dtype), _fp8_max(fp8_dtype))
    byte_values = lax.bitcast_convert_type(values.astype(fp8_dtype), jnp.uint8)
    latent_dim = cache_k_nope.shape[-1]
    latent = byte_values[..., :latent_dim]
    rope = jnp.concatenate(
        (byte_values[..., latent_dim:], _as_byte_view(scales, jnp.float32, 4)), axis=-1
    )

    def pack(x, shape):
        padding = math.prod(shape) - x.shape[-1]
        x = jnp.pad(x, ((0, 0),) * (x.ndim - 1) + ((0, padding),))
        return x.reshape(*x.shape[:-1], *shape)

    return SplitFP8KVRows(pack(latent, layout.latent_shape), pack(rope, (layout.rope_shape[-1],)))


def group_split_fp8_kv_rows(rows: SplitFP8KVRows) -> SplitFP8KVCache:
    """Build an unpaged cache from consecutive rows (fixtures and benchmarks)."""
    if rows.latent.ndim != 3 or rows.rope.ndim != 2 or rows.latent.shape[0] != rows.rope.shape[0]:
        raise ValueError("Expected per-token latent [S, 4, D] and RoPE [S, R] rows.")
    if rows.rope.shape[0] % 4:
        raise ValueError("Grouped RoPE storage requires a token count divisible by four.")
    return SplitFP8KVCache(rows.latent, rows.rope.reshape(-1, 4, rows.rope.shape[-1]))


def flatten_split_fp8_tile(
    cache: SplitFP8KVCache | SplitFP8KVRows | GatheredFP8KVGroups, latent_dim: int
) -> jnp.ndarray:
    """Join loaded VMEM payloads, excluding any latent padding before RoPE."""
    latent = cache.latent.reshape(*cache.latent.shape[:-2], -1)[..., :latent_dim]
    rope = cache.rope
    if isinstance(cache, GatheredFP8KVGroups):
        from jax.experimental.pallas import tpu as pltpu

        # Mosaic packs the four rows on the second-minor axis into uint32.
        # Select one byte per column in TC VMEM, including that token's scale.
        words = pltpu.bitcast(rope, jnp.uint32).reshape(*rope.shape[:-2], rope.shape[-1])
        shift = cache.rope_lanes[..., None].astype(jnp.uint32) * 8
        rope = ((words >> shift) & jnp.uint32(255)).astype(jnp.uint8)
    else:
        # Persistent groups are flattened only after loading their VMEM tile.
        rope = rope.reshape(*latent.shape[:-1], rope.shape[-1])
    return jnp.concatenate((latent, rope), axis=-1)


def dequantize_split_fp8_kv_cache(
    cache: SplitFP8KVCache | SplitFP8KVRows,
    *,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    fp8_dtype=jnp.float8_e4m3fn,
    output_dtype=jnp.bfloat16,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if any(part.dtype != jnp.uint8 for part in cache):
        raise TypeError("Split FP8 KV storage must use uint8 payloads.")
    _fp8_max(fp8_dtype)
    flat = flatten_split_fp8_tile(cache, kv_lora_rank)
    quantized_dim = kv_lora_rank + qk_rope_head_dim
    quantized = lax.bitcast_convert_type(flat[..., :quantized_dim], fp8_dtype).astype(jnp.float32)
    scale = lax.bitcast_convert_type(
        flat[..., quantized_dim : quantized_dim + 4].reshape(*flat.shape[:-1], 1, 4), jnp.float32
    )
    values = (quantized * scale).astype(output_dtype)
    return values[..., :kv_lora_rank], values[..., kv_lora_rank:quantized_dim]
