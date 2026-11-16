# SPDX-License-Identifier: Apache-2.0
# Adapted from primatrix-glm/sglang-jax-private (Apache-2.0) at
# c948bdda95208e68a474c8088937347d08fe8937:
#   python/sgl_jax/srt/layers/attention/dsa_indexer_ops.py
#     (project_index_keys_update_cache_and_select: project_keys_xla, the aligned
#      check, target pages, lax.cond), FP8 per-channel branch only;
#   python/sgl_jax/srt/layers/attention/dsa_cache_ops.py (scatter_paged_cache, 3-D);
#   python/sgl_jax/srt/kernels/quantized_matmul/kernel.py (_per_channel_dot, W8A16).
"""Source-contract index-key prologue: hidden -> BF16 index-key cache, page 64.

The source computation is kept as is: W8A16 ``wk`` dot (f32 accumulation,
per-channel scale after the dot, bf16 output), LayerNorm eps 1e-5 with bf16
parameters, interleaved rope on dims [:64], f32 Hadamard ``key @ H``, BF16 cache
``[P, 64, 128]``. Page-aligned extend batches run the source Pallas kernel
(``dsa_source_key_projection``, verbatim); every other batch runs the source's
XLA chain. Framework adaptation only: no ``shard_map`` (single device), and the
XLA fallback always uses the plain per-channel dot, not the source's optional
tuned per-channel Pallas matmul.
"""

import jax
import jax.numpy as jnp
from jax import lax

from sglang_tpu.attention.pallas.dsa_source_key_projection import (
    fused_fp8_k_projection_scatter_pallas,
)

PAGE_SIZE = 64
INDEX_DIM = 128
ROPE_DIM = 64


def hadamard_128() -> jax.Array:
    """Orthonormal Sylvester Hadamard H128 * 128^-0.5, f32 (the source's ``key @ H``)."""
    h = jnp.ones((1, 1), jnp.float32)
    while h.shape[0] < INDEX_DIM:
        h = jnp.block([[h, h], [h, -h]])
    return h * (INDEX_DIM**-0.5)


def project_keys_xla(hidden, wk, wk_scale, norm_weight, norm_bias, rope_cos, rope_sin, hadamard):
    """The source's ``project_keys_xla`` (FP8 ``wk`` [128, K], W8A16)."""
    key = lax.dot_general(
        hidden, wk, dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    key = (key * wk_scale[None, :]).astype(hidden.dtype)
    mean = jnp.mean(key, axis=-1, keepdims=True)
    variance = jnp.var(key, axis=-1, keepdims=True)
    key = (key - mean) / jnp.sqrt(variance + 1e-5)
    key = key * norm_weight + norm_bias
    key_rope = key[:, :ROPE_DIM]
    even, odd = key_rope[:, ::2], key_rope[:, 1::2]
    rotated = jnp.stack(
        (even * rope_cos - odd * rope_sin, odd * rope_cos + even * rope_sin), axis=-1
    ).reshape(key_rope.shape)
    key = key.at[:, :ROPE_DIM].set(rotated)
    return jnp.einsum("td,de->te", key, hadamard)


def scatter_paged_cache(cache, new_tokens, seq_lens, page_indices, cu_q_lens, cu_kv_lens):
    """The source's 3-D ``scatter_paged_cache``; padded tokens go to scratch page 0."""
    page_size = cache.shape[1]
    num_tokens, num_seqs = new_tokens.shape[0], seq_lens.shape[0]
    token_id = jnp.arange(num_tokens)
    seq_id = jnp.clip(jnp.searchsorted(cu_q_lens[1:], token_id, side="right"), 0, num_seqs - 1)
    q_start, q_end = cu_q_lens[seq_id], cu_q_lens[seq_id + 1]
    kv_len = seq_lens[seq_id]
    abs_pos = jnp.maximum(kv_len - (q_end - q_start) + (token_id - q_start), 0)
    valid = (token_id >= q_start) & (token_id < q_end) & (kv_len > 0)
    page = page_indices[cu_kv_lens[seq_id] // page_size + abs_pos // page_size]
    safe_page = jnp.where(valid, page, 0)
    safe_offset = jnp.where(valid, abs_pos % page_size, 0)
    return cache.at[safe_page, safe_offset].set(new_tokens.astype(cache.dtype))


def aligned_target_pages(num_tokens, seq_lens, page_indices, cu_q_lens, cu_kv_lens, num_pages):
    """Physical page per 64-token chunk, and whether the batch fills whole fresh pages."""
    num_token_pages = num_tokens // PAGE_SIZE
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    prefix_lens = seq_lens - q_lens
    token_starts = jnp.arange(num_token_pages, dtype=jnp.int32) * PAGE_SIZE
    seq_ids = jnp.clip(
        jnp.searchsorted(cu_q_lens[1:], token_starts, side="right"), 0, seq_lens.shape[0] - 1
    )
    logical_starts = prefix_lens[seq_ids] + token_starts - cu_q_lens[seq_ids]
    table_idx = cu_kv_lens[seq_ids] // PAGE_SIZE + logical_starts // PAGE_SIZE
    safe_idx = jnp.clip(table_idx, 0, page_indices.shape[0] - 1)
    target_pages = page_indices[safe_idx].astype(jnp.int32)
    aligned = (
        (cu_q_lens[0] == 0)
        & (cu_q_lens[-1] == num_tokens)
        & jnp.all(q_lens >= 0)
        & jnp.all(q_lens % PAGE_SIZE == 0)
        & jnp.all(prefix_lens >= 0)
        & jnp.all(prefix_lens % PAGE_SIZE == 0)
        & jnp.all(table_idx == safe_idx)
        & jnp.all(target_pages >= 0)
        & jnp.all(target_pages < num_pages)
    )
    return target_pages, aligned


def decode_token_slots(num_tokens, seq_lens, page_indices, cu_q_lens, cu_kv_lens):
    """The source's per-token flat slot ``page * 64 + offset``; -1 for padding tokens."""
    token_id = jnp.arange(num_tokens, dtype=jnp.int32)
    seq_id = jnp.clip(jnp.searchsorted(cu_q_lens[1:], token_id, side="right"), 0, seq_lens.shape[0] - 1)
    q_start, q_end = cu_q_lens[seq_id], cu_q_lens[seq_id + 1]
    kv_len = seq_lens[seq_id]
    position = jnp.maximum(kv_len - (q_end - q_start) + token_id - q_start, 0)
    page = page_indices[cu_kv_lens[seq_id] // PAGE_SIZE + position // PAGE_SIZE]
    valid = (token_id >= q_start) & (token_id < q_end) & (kv_len > 0)
    return jnp.where(valid, page * PAGE_SIZE + position % PAGE_SIZE, -1).astype(jnp.int32)


def store_index_keys_bf16(
    cache, hidden, wk, wk_scale, norm_weight, norm_bias, rope_cos, rope_sin, hadamard,
    seq_lens, page_indices, cu_q_lens, cu_kv_lens, *, one_token_per_seq=False, interpret=False,
):
    """Write this step's index keys into ``cache`` (bf16 [P, 64, 128]).

    ``wk`` is f8e4m3fn [128, K] with an f32 [128] channel scale; ``rope_cos`` /
    ``rope_sin`` are [T, 32] per token. ``one_token_per_seq`` (decode) runs the
    source kernel in ``token_slots`` mode, one read-modify-write per token, as
    the source does for an FP8 ``wk`` on TPU; extend batches use the aligned
    kernel or the XLA chain.
    """
    if cache.ndim != 3 or cache.shape[1:] != (PAGE_SIZE, INDEX_DIM) or cache.dtype != jnp.bfloat16:
        raise ValueError(f"expected a bf16 [P, {PAGE_SIZE}, {INDEX_DIM}] cache, got {cache.shape} {cache.dtype}")
    # No backend check, unlike the source: the torch op exports on the CPU backend
    # (JAX_PLATFORMS=cpu) for a TPU target, so default_backend() is not "tpu" there.
    if one_token_per_seq:
        slots = decode_token_slots(hidden.shape[0], seq_lens, page_indices, cu_q_lens, cu_kv_lens)
        return fused_fp8_k_projection_scatter_pallas(
            hidden, wk, wk_scale, norm_weight, norm_bias, rope_cos, rope_sin, hadamard,
            cache, slots, page_size=PAGE_SIZE, interpret=interpret, token_slots=True,
        )

    def xla(current):
        key = project_keys_xla(hidden, wk, wk_scale, norm_weight, norm_bias, rope_cos, rope_sin, hadamard)
        return scatter_paged_cache(current, key, seq_lens, page_indices, cu_q_lens, cu_kv_lens)

    num_tokens = hidden.shape[0]
    if num_tokens < PAGE_SIZE or num_tokens % PAGE_SIZE:
        return xla(cache)
    target_pages, aligned = aligned_target_pages(
        num_tokens, seq_lens, page_indices, cu_q_lens, cu_kv_lens, cache.shape[0]
    )

    def fused(current):
        return fused_fp8_k_projection_scatter_pallas(
            hidden, wk, wk_scale, norm_weight, norm_bias, rope_cos, rope_sin, hadamard,
            current, target_pages, page_size=PAGE_SIZE, interpret=interpret,
        )

    return lax.cond(aligned, fused, xla, cache)
