# SPDX-License-Identifier: Apache-2.0
# Copied verbatim from primatrix-glm/sglang-jax-private (Apache-2.0) at
# c948bdda95208e68a474c8088937347d08fe8937:
#   python/sgl_jax/srt/kernels/dsa/fused_q_rope_hadamard.py
# Everything below the next line is the source file byte for byte (sha256 d8d8dedb...).
# ---- source ----
"""Experimental prefill indexer Q tail; projection remains native W8A16."""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


@functools.partial(jax.jit, static_argnames=("token_tile", "interpret"))
def fused_q_rope_hadamard(query, cos, sin, hadamard, *, token_tile=16, interpret=False):
    """BF16 Q[T,32,128], expanded signed BF16 coefficients[T,128] -> FP32 Q.

    Use exactly the expanded coefficients consumed by PackedPrefillRotaryEmbedding.
    Retain BF16 rotation rounding and the dense FP32 Hadamard contraction.
    """
    tokens, heads, width = query.shape
    if heads != 32 or width != 128 or tokens % token_tile or token_tile % 8:
        raise ValueError("expected Q[T,32,128] with T divisible by token_tile (a multiple of 8)")
    if query.dtype != jnp.bfloat16 or cos.dtype != query.dtype or sin.dtype != query.dtype:
        raise ValueError("Q and RoPE coefficients must be BF16")
    if cos.shape != (tokens, 128) or sin.shape != cos.shape:
        raise ValueError("expected expanded coefficients[T,128]")
    if hadamard.shape != (128, 128) or hadamard.dtype != jnp.float32:
        raise ValueError("expected FP32 Hadamard[128,128]")

    def kernel(q, c, s, h, out):
        x = q[...].reshape(token_tile * heads, width)
        cc = jnp.broadcast_to(c[...][:, None, :], (token_tile, heads, width)).reshape(x.shape)
        ss = jnp.broadcast_to(s[...][:, None, :], (token_tile, heads, width)).reshape(x.shape)
        lanes = jnp.arange(width)[None, :]
        partner = jnp.where(lanes % 2 == 0, pltpu.roll(x, 127, axis=1), pltpu.roll(x, 1, axis=1))
        rotated = (x * cc + partner * ss).astype(jnp.bfloat16)
        full_q = jnp.where(lanes < 64, rotated, x)
        y = jax.lax.dot_general(
            full_q.astype(jnp.float32),
            h[...],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        out[...] = y.reshape(token_tile, heads, width)

    qs = pl.BlockSpec((token_tile, heads, width), lambda t: (t, 0, 0))
    cs = pl.BlockSpec((token_tile, width), lambda t: (t, 0))
    hs = pl.BlockSpec((width, width), lambda t: (0, 0))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(query.shape, jnp.float32),
        grid=(tokens // token_tile,),
        in_specs=(qs, cs, cs, hs),
        out_specs=qs,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
        name="fused_indexer_q_rope_hadamard",
    )(query, cos, sin, hadamard)
