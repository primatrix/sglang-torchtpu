"""A page-128 index-key cache whose every slot holds a valid record.

Written by the repo's own ``quantize_fp8_ue8m0`` + ``_store_keys`` (one request
spanning every page), so the bytes are the ones serving writes. Runs on
whatever JAX backend the process selected; the torch A/B calls it with JAX on
the CPU and copies the result to the TPU.
"""

import jax.numpy as jnp
import numpy as np

from prologue_common import D, prefix_keys


def valid_prefix_cache(num_pages: int, seed: int, page: int = 128) -> np.ndarray:
    from sglang_tpu.attention.pallas import dsa_index_topk as dit
    from sglang_tpu.attention.pallas.util import quantize_fp8_ue8m0

    tokens = num_pages * page
    k_fp8, _, k_byte = quantize_fp8_ue8m0(jnp.asarray(prefix_keys(tokens, seed)).astype(jnp.bfloat16))
    cache = jnp.zeros(dit.index_key_cache_shape(num_pages, page, D), jnp.uint8)
    cache = dit._store_keys(
        cache, k_fp8, k_byte,
        jnp.array([tokens], jnp.int32),
        jnp.arange(num_pages, dtype=jnp.int32),
        jnp.array([0, tokens], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32),
    )
    return np.asarray(cache)
