# SPDX-License-Identifier: Apache-2.0
"""Source-contract DSA indexer (S1-S4) on the framework's page-128 page table.

Chains the migrated source modules: the BF16 index-key store (M1-S,
``dsa_bf16_key_store``), the query rope + Hadamard tail (M2-S,
``dsa_bf16_query``) and the source scorer + radix top-k (M3-S,
``source_dsa``). The source computation is unchanged; only the page geometry
and the output format are adapted.

Framework adaptation (not in the source):

* Page geometry. The framework allocates 128-token pages; the source kernels
  need 64. The cache is bf16 ``[P, 128, 128]``, which is ``[2P, 64, 128]`` in
  the same memory, and framework page ``p`` is source pages ``2p`` and
  ``2p + 1``. A token's flat slot ``p * 128 + o`` is the same number either way.
* ``cu_kv_lens``. The framework's page table has a fixed number of pages per
  request row, so the token offset of row ``s`` is ``s * pages_per_row * 128``.
* Output format. ``physical=False`` returns sequence-local logical positions,
  which the target sparse attention reads. ``physical=True`` is the source
  deployment (``output_physical_slots=True``): flat cache slots for the
  migrated S6 attention, which then only validates them. Slot ``p * 128 + o``
  of a framework page equals the source's page-64 slot.

Head weights go in as ``weights_proj`` produces them, without the target's
``softmax_scale * n_heads ** -0.5``: the source applies no factor. A positive
constant on every head weight does not change the top-k order.
"""

import jax.numpy as jnp

from sglang_tpu.attention.pallas.dsa_bf16_key_store import store_index_keys_bf16
from sglang_tpu.attention.pallas.dsa_bf16_query import index_query_tail
from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.indexer import (
    compute_scores_and_select_topk_indices,
)

FRAMEWORK_PAGE = 128
SOURCE_PAGE = 64
SPLIT = FRAMEWORK_PAGE // SOURCE_PAGE
INDEX_DIM = 128


def source_page_table(page_indices):
    """Framework pages ``[p, ...]`` -> source pages ``[2p, 2p + 1, ...]``."""
    p = page_indices.astype(jnp.int32)
    return jnp.stack([p * SPLIT + j for j in range(SPLIT)], axis=-1).reshape(-1)


def source_cache_view(cache):
    """bf16 ``[P, 128, 128]`` -> ``[2P, 64, 128]`` (same memory, row-major)."""
    if cache.ndim != 3 or cache.shape[1:] != (FRAMEWORK_PAGE, INDEX_DIM) or cache.dtype != jnp.bfloat16:
        raise ValueError(f"expected a bf16 [P, {FRAMEWORK_PAGE}, {INDEX_DIM}] cache, got {cache.shape} {cache.dtype}")
    return cache.reshape(cache.shape[0] * SPLIT, SOURCE_PAGE, INDEX_DIM)


def bf16_indexer_step(
    cache, hidden, wk, wk_scale, norm_weight, norm_bias, query, rope_cos, rope_sin,
    head_weights, kv_lens, page_indices, cu_q_lens, distribution, hadamard=None,
    *, topk, prefill, decode, physical=False, topk_impl="radix", interpret=False,
):
    """Store this step's index keys, then pick each query's top-k keys.

    ``query`` is the bf16 ``wq_b`` output ``[T, 32, 128]``; ``rope_cos`` /
    ``rope_sin`` the bf16 ``[T, 32]`` phases of each token's position;
    ``kv_lens`` / ``page_indices`` / ``cu_q_lens`` / ``distribution`` the
    framework's metadata (page 128, fixed pages per request row). ``prefill``
    selects the source's fused query tail, ``decode`` its one-token-per-request
    store and scorer. Returns ``(topk_indices, cache)``: ``i32[T, topk]``
    sequence-local positions (flat cache slots when ``physical``), ``-1``
    padded, and the updated cache.
    """
    num_seqs = kv_lens.shape[0]
    pages_per_row = page_indices.shape[0] // num_seqs
    table = source_page_table(page_indices)
    cu_kv_lens = jnp.arange(num_seqs + 1, dtype=jnp.int32) * (pages_per_row * FRAMEWORK_PAGE)
    view = source_cache_view(cache)

    view = store_index_keys_bf16(
        view, hidden, wk, wk_scale, norm_weight, norm_bias, rope_cos, rope_sin,
        hadamard if hadamard is not None else _hadamard(), kv_lens, table, cu_q_lens,
        cu_kv_lens, one_token_per_seq=decode, interpret=interpret,
    )
    q_idx = index_query_tail(
        query, rope_cos, rope_sin, hadamard, prefill=prefill, interpret=interpret
    )
    topk_indices = compute_scores_and_select_topk_indices(
        q_idx, head_weights, view, kv_lens, table, cu_q_lens, cu_kv_lens, distribution,
        k=topk, pages_per_seq=pages_per_row * SPLIT, one_token_per_seq=decode,
        topk_impl=topk_impl, output_physical_slots=physical,
    )
    return topk_indices, view.reshape(cache.shape)


def _hadamard():
    from sglang_tpu.attention.pallas.dsa_bf16_key_store import hadamard_128

    return hadamard_128()
