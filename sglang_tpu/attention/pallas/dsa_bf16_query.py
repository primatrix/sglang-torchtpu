# SPDX-License-Identifier: Apache-2.0
# Adapted from primatrix-glm/sglang-jax-private (Apache-2.0) at
# c948bdda95208e68a474c8088937347d08fe8937:
#   python/sgl_jax/srt/models/glm5_moe.py (GlmDsaIndexer._project_query_and_weights, Q tail);
#   python/sgl_jax/srt/layers/packed_rope.py (_expanded_coefficients);
#   python/sgl_jax/srt/layers/embeddings.py (apply_rotary_emb, GPT-J branch).
"""Source-contract indexer query tail: Q[T, 32, 128] bf16 -> rope + Hadamard, f32.

The source computation is kept as is: interleaved (GPT-J) rope on dims [:64] in
bf16, then the dense f32 Hadamard ``q @ H``, output f32. Prefill batches whose
token count is a multiple of 64 run the source Pallas kernel
(``dsa_source_q_rope_hadamard``, verbatim) on the expanded coefficients; every
other batch runs the source's ``apply_rotary_emb`` + einsum. Framework
adaptation only: no ``shard_map`` (single device, so the source's
``64 * data`` token condition is ``64``), and no ``PackedPrefillRotaryEmbedding``
object: the caller passes the bf16 cos / sin of each token's position.
"""

import jax.numpy as jnp

from sglang_tpu.attention.pallas.dsa_bf16_key_store import hadamard_128
from sglang_tpu.attention.pallas.dsa_source_q_rope_hadamard import fused_q_rope_hadamard

HEADS, HEAD_DIM, ROPE_DIM = 32, 128, 64
TOKEN_TILE = 64  # the source's call site


def expanded_coefficients(cos, sin):
    """The source's ``_expanded_coefficients``: bf16 [T, 32] -> signed bf16 [T, 128]."""
    cos, sin = cos.astype(jnp.bfloat16), sin.astype(jnp.bfloat16)
    return (
        jnp.tile(jnp.repeat(cos, 2, axis=-1), (1, 2)),
        jnp.tile(jnp.stack((-sin, sin), -1).reshape(-1, ROPE_DIM), (1, 2)),
    )


def apply_rotary_emb_gptj(x, cos, sin):
    """The source's ``apply_rotary_emb`` (non-neox): x [T, H, 64], cos / sin [T, 32]."""
    cos = jnp.expand_dims(cos, -2).astype(x.dtype)
    sin = jnp.expand_dims(sin, -2).astype(x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = jnp.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), axis=-1)
    return out.reshape(*out.shape[:-2], -1)


def query_tail_xla(query, cos, sin, hadamard):
    q_rope = apply_rotary_emb_gptj(query[:, :, :ROPE_DIM], cos, sin)
    query = query.at[:, :, :ROPE_DIM].set(q_rope)
    return jnp.einsum("thd,de->the", query, hadamard)


def index_query_tail(query, cos, sin, hadamard=None, *, prefill, interpret=False):
    """Rope + Hadamard of the indexer query, as the source computes it; returns f32.

    ``query`` is the bf16 ``wq_b`` output [T, 32, 128]; ``cos`` / ``sin`` are the
    bf16 rope phases [T, 32] of each token's position.
    """
    if query.shape[1:] != (HEADS, HEAD_DIM) or query.dtype != jnp.bfloat16:
        raise ValueError(f"expected bf16 Q[T, {HEADS}, {HEAD_DIM}], got {query.shape} {query.dtype}")
    hadamard = hadamard_128() if hadamard is None else hadamard
    cos, sin = cos.astype(jnp.bfloat16), sin.astype(jnp.bfloat16)
    if prefill and query.shape[0] > 0 and query.shape[0] % TOKEN_TILE == 0:
        c, s = expanded_coefficients(cos, sin)
        return fused_q_rope_hadamard(query, c, s, hadamard, token_tile=TOKEN_TILE, interpret=interpret)
    return query_tail_xla(query, cos, sin, hadamard)
