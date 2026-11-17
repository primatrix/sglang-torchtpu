# SPDX-License-Identifier: Apache-2.0
"""Source-contract DSA KV write + exact sparse attention (S5-S6), split FP8 KV.

Chains the migrated source modules (``source_dsa``) the way the source
``DSASparseAttentionBackend._run_exact`` does for the FP8 delivery
(``--kv-cache-dtype fp8_e4m3``, ``--dsa-kv-cache-layout split``,
``dsa_attention_precision="fp8-qk-pv"``):

1. quantize this step's latent + k_rope with one shared f32 scale per token and
   scatter them into the split cache (``_scatter_fused_kv_paged``);
2. map the indexer's sequence-local top-k positions to physical cache rows
   (``_logical_topk_to_physical_slots``), or, when the indexer already emits
   flat cache slots (the source deployment, ``topk_indices_are_physical``),
   only validate them (the backend's ``PhysicalTopKValidation``);
3. resolve the execution plan (``DsaConfig.resolve``), pad the queries to the
   SparseCore microbatch and run ``sparse_core_tensor_core_dsa``.

The source computation is unchanged. Framework adaptation (not in the source):
at page 64 (the PR 2198 baseline) the cache is the source's own layout; at
page 128 it is viewed as page 64 (framework page ``p`` = source pages ``2p``,
``2p + 1``), as ``dsa_bf16_indexer`` does for the index cache. At page 128, Split latent ``[P, 128, 4, 128]`` and RoPE ``[P, 32, 4, 128]`` are
``[2P, 64, 4, 128]`` and ``[2P, 16, 4, 128]`` in the same memory.
"""

import jax
import jax.numpy as jnp

from sglang_tpu.attention.pallas.source_dsa.configs.dsa_config import DsaConfig
from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.exact_attention import (
    sparse_core_tensor_core_dsa,
)
from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.fp8_kv_cache import (
    SplitFP8KVCache,
    cache_primary,
    map_cache_parts,
)
from sglang_tpu.attention.pallas.source_dsa.layers.attention.dsa_sparse_backend_helpers import (
    _logical_topk_to_physical_slots,
    _pad_first_axis,
    _scatter_fused_kv_paged,
)

FRAMEWORK_PAGE = 128
SOURCE_PAGE = 64
SPLIT = FRAMEWORK_PAGE // SOURCE_PAGE
KV_LORA_RANK = 512
FP8 = jnp.float8_e4m3fn


def source_step(
    cache, q_latent, q_rope, kv_c, k_pe, topk, seq_lens, page_indices, cu_q_lens, cu_kv_lens,
    sm_scale, *, allow_fp8_q, is_prefill, topk_is_physical=False, precision="fp8-qk-pv",
    tiles=None,
):
    """The source ``_run`` body at page 64: returns ``(out [T, H, 512], cache)``."""
    page_size = cache_primary(cache).shape[1]
    cache = _scatter_fused_kv_paged(
        cache, kv_c, k_pe, seq_lens, page_indices, cu_q_lens, cu_kv_lens,
        kv_lora_rank=KV_LORA_RANK, fp8_dtype=FP8,
    )
    if topk_is_physical:
        # dsa_sparse_backend.py PhysicalTopKValidation (source c948bdd, lines 212-216).
        valid = (topk >= 0) & (topk < cache_primary(cache).shape[0] * page_size)
        slots = jnp.where(valid, topk, jnp.int32(0))
        counts = jnp.sum(valid, axis=1, dtype=jnp.int32)
    else:
        slots, counts = _logical_topk_to_physical_slots(
            topk, seq_lens, page_indices, cu_q_lens, cu_kv_lens, page_size
        )
    plan = DsaConfig(precision=precision).resolve(
        q_latent.shape[0], eligible=allow_fp8_q, is_prefill=is_prefill, fp8_kv=True
    )
    if plan.cache_read_mode != "packed":
        raise ValueError(f"expanded cache reads are not migrated (plan {plan})")
    # ``tiles`` (bq_sparse, bq, b_topk) overrides the plan's tiles; for tuning only.
    bq_sparse, bq, b_topk = tiles if tiles is not None else plan.tiles
    flat = map_cache_parts(lambda part: part.reshape(-1, *part.shape[2:]), cache)
    out = sparse_core_tensor_core_dsa(
        _pad_first_axis(q_latent, bq_sparse), _pad_first_axis(q_rope, bq_sparse), flat,
        _pad_first_axis(slots, bq_sparse), _pad_first_axis(counts, bq_sparse), sm_scale,
        bq_sparse=bq_sparse, bq=bq, b_topk=b_topk, fp8_dtype=FP8,
        fp8_q=plan.fp8_q, fp8_pv=plan.fp8_pv,
        **({} if plan.pipeline == "overlap" else {"split_gather_buffers": plan.pipeline == "split"}),
    )
    return out[: q_latent.shape[0]], cache


def source_page_table(page_indices, split=SPLIT):
    """Framework pages ``[p, ...]`` -> source pages ``[2p, 2p + 1, ...]`` (page 128)."""
    p = page_indices.astype(jnp.int32)
    if split == 1:
        return p
    return jnp.stack([p * split + j for j in range(split)], axis=-1).reshape(-1)


def fp8_attention_step(
    latent, rope, q_latent, q_rope, kv_c, k_pe, topk, kv_lens, page_indices, cu_q_lens,
    sm_scale, *, allow_fp8_q, is_prefill, topk_is_physical=False, tiles=None,
):
    """Framework entry: split u8 latent ``[P, page, 4, 128]`` + RoPE ``[P, page // 4, 4, 128]``.

    ``page`` is 64 (the source's own page, run as is) or 128 (viewed as two
    64-token source pages). ``topk`` is the indexer's sequence-local rows (-1
    padded), or its flat cache slots when ``topk_is_physical``; the page table
    has a fixed number of pages per request row. Returns ``(out, latent, rope)``.
    """
    page = latent.shape[1]
    if page not in (SOURCE_PAGE, FRAMEWORK_PAGE) or latent.shape[2:] != (4, 128) \
            or rope.shape[1:] != (page // 4, 4, 128):
        raise ValueError(f"expected split FP8 KV at page 64 or 128, got {latent.shape} {rope.shape}")
    split = page // SOURCE_PAGE
    num_seqs = kv_lens.shape[0]
    pages_per_row = page_indices.shape[0] // num_seqs
    cu_kv_lens = jnp.arange(num_seqs + 1, dtype=jnp.int32) * (pages_per_row * page)
    view = SplitFP8KVCache(
        latent.reshape(-1, SOURCE_PAGE, 4, 128), rope.reshape(-1, SOURCE_PAGE // 4, 4, 128)
    )
    out, view = source_step(
        view, q_latent, q_rope, kv_c, k_pe, topk, kv_lens, source_page_table(page_indices, split),
        cu_q_lens, cu_kv_lens, sm_scale, allow_fp8_q=allow_fp8_q, is_prefill=is_prefill,
        topk_is_physical=topk_is_physical, tiles=tiles,
    )
    return out, view.latent.reshape(latent.shape), view.rope.reshape(rope.shape)


def jit_source_step(**static):
    return jax.jit(lambda *a: source_step(*a, **static))
