# SPDX-License-Identifier: Apache-2.0
# Verbatim copy of primatrix-glm/sglang-jax-private (Apache-2.0)
# python/sgl_jax/srt/kernels/dsa/fused_fp8_k_projection_scatter.py at
# c948bdda95208e68a474c8088937347d08fe8937, sha256
# f7a8cef892275daa98181a0cb10b62450a48d654b40d0cd4734f94ee8ad08879.
# Everything below these header lines is byte-identical; do not edit it here.
"""Per-channel W8A16 Indexer K projection through paged cache DMA.

Matches the native projection contract: mixed BF16/FP8 dot, FP32 accumulation,
per-output scale after reduction, BF16 projection output. Q and score weights
remain outside. Variance and sqrt retain FP32 precision before the BF16 denominator cast.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _fused_fp8_k_projection_scatter_kernel(
    target_pages_ref,  # i32[num_token_pages], SMEM
    hidden_hbm_ref,  # bf16[T, hidden_dim], HBM
    wk_vmem_ref,  # f8e4m3fn[128, hidden_dim], VMEM
    scale_vmem_ref,  # f32[1,128], applied after FP32 accumulation
    norm_weight_vmem_ref,  # bf16[128], VMEM
    norm_bias_vmem_ref,  # bf16[128], VMEM
    rope_cos_vmem_ref,  # bf16[T, 32], VMEM
    rope_sin_vmem_ref,  # bf16[T, 32], VMEM
    hadamard_vmem_ref,  # f32[128, 128], VMEM
    _cache_hbm_ref,  # bf16[P, page_size, 128], HBM (aliased input)
    updated_cache_hbm_ref,  # aliased output
    hidden_x2_vmem_ref,  # bf16[2, page_size, hidden_dim]
    cache_page_x2_vmem_ref,  # bf16[2, page_size, 128]
    hidden_dma_sems,  # DMA semaphore[2]
    cache_dma_sems,  # DMA semaphore[2]
    update_page_ref,
    update_sem,
    *,
    token_slots: bool,
    num_token_pages: int,
    page_size: int,
):
    """Project one token page at a time and asynchronously write cache pages."""

    def fetch_hidden(page_id, bank, *, wait: bool):
        copy = pltpu.make_async_copy(
            hidden_hbm_ref.at[pl.ds(page_id * page_size, page_size)],
            hidden_x2_vmem_ref.at[bank],
            hidden_dma_sems.at[bank],
        )
        if wait:
            copy.wait()
        else:
            copy.start()

    def write_cache(page_id, bank, *, wait: bool):
        if token_slots:
            if wait:
                return  # Partial-page writes below complete synchronously.

            def update_row(row, _):
                slot = target_pages_ref[page_id * page_size + row]

                @pl.when(slot >= 0)
                def write_row():
                    # Serial read/modify/write also handles two tokens sharing
                    # a physical page without overwriting the first update.
                    page = slot // page_size
                    read = pltpu.make_async_copy(
                        updated_cache_hbm_ref.at[page], update_page_ref, update_sem
                    )
                    read.start()
                    read.wait()
                    row_ids = jnp.arange(page_size)[:, None]
                    value = jnp.sum(
                        jnp.where(
                            row_ids == row, cache_page_x2_vmem_ref[bank].astype(jnp.float32), 0
                        ),
                        axis=0,
                        keepdims=True,
                    ).astype(update_page_ref.dtype)
                    update_page_ref[...] = jnp.where(
                        row_ids == slot % page_size, value, update_page_ref[...]
                    )
                    write = pltpu.make_async_copy(
                        update_page_ref, updated_cache_hbm_ref.at[page], update_sem
                    )
                    write.start()
                    write.wait()

                return None

            lax.fori_loop(0, page_size, update_row, None)
            return

        physical_page = target_pages_ref[page_id]
        copy = pltpu.make_async_copy(
            cache_page_x2_vmem_ref.at[bank],
            updated_cache_hbm_ref.at[physical_page],
            cache_dma_sems.at[bank],
        )
        if wait:
            copy.wait()
        else:
            copy.start()

    def project_page(page_id, bank):
        hidden = hidden_x2_vmem_ref[bank]
        acc = lax.dot_general(
            hidden,
            wk_vmem_ref[...],
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        key = (acc * scale_vmem_ref[...]).astype(jnp.bfloat16)
        # Keep the variance-to-sqrt chain in FP32: early BF16 rounding and
        # BF16 sqrt differ from the native fused Norm on TPU.
        mean = jnp.mean(key, axis=-1, keepdims=True)
        variance = jnp.var(key.astype(jnp.float32), axis=-1, keepdims=True)
        denominator = jnp.sqrt(variance + 1e-5).astype(jnp.bfloat16)
        key = ((key - mean) / denominator) * norm_weight_vmem_ref[...] + norm_bias_vmem_ref[...]

        token_start = page_id * page_size
        cos = rope_cos_vmem_ref[pl.ds(token_start, page_size)]
        sin = rope_sin_vmem_ref[pl.ds(token_start, page_size)]
        key_rope = key[:, :64]
        # Mosaic rejects both the strided ``[:, ::2]`` gather and a
        # [page, 64] -> [page, 32, 2] shape cast, so expand the GPT-J pairs at
        # compile time into unit-width slices.
        rotated_columns = []
        for pair_id in range(32):
            even = lax.slice_in_dim(key_rope, 2 * pair_id, 2 * pair_id + 1, axis=1)
            odd = lax.slice_in_dim(key_rope, 2 * pair_id + 1, 2 * pair_id + 2, axis=1)
            cos_i = lax.slice_in_dim(cos, pair_id, pair_id + 1, axis=1)
            sin_i = lax.slice_in_dim(sin, pair_id, pair_id + 1, axis=1)
            rotated_columns.extend((even * cos_i - odd * sin_i, odd * cos_i + even * sin_i))
        rotated = jnp.concatenate(rotated_columns, axis=1)
        key = jnp.concatenate((rotated, key[:, 64:]), axis=-1)

        # Match the existing dense FP32 Hadamard einsum before the cache cast.
        key = lax.dot_general(
            key.astype(jnp.float32),
            hadamard_vmem_ref[...],
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        cache_page_x2_vmem_ref[bank] = key.astype(updated_cache_hbm_ref.dtype)

    fetch_hidden(0, 0, wait=False)

    def page_body(page_id, _):
        bank = page_id & 1
        fetch_hidden(page_id, bank, wait=True)

        @pl.when(page_id + 1 < num_token_pages)
        def prefetch_next_hidden():
            fetch_hidden(page_id + 1, 1 - bank, wait=False)

        @pl.when(page_id >= 2)
        def wait_old_cache_write():
            write_cache(page_id - 2, bank, wait=True)

        project_page(page_id, bank)
        write_cache(page_id, bank, wait=False)
        return None

    lax.fori_loop(0, num_token_pages, page_body, None, unroll=False)

    @pl.when(num_token_pages >= 2)
    def drain_penultimate():
        page_id = num_token_pages - 2
        write_cache(page_id, page_id & 1, wait=True)

    page_id = num_token_pages - 1
    write_cache(page_id, page_id & 1, wait=True)


@functools.partial(
    jax.jit,
    static_argnames=("page_size", "interpret", "token_slots"),
)
def fused_fp8_k_projection_scatter_pallas(
    hidden_states: jax.Array,
    wk: jax.Array,
    scale: jax.Array,
    norm_weight: jax.Array,
    norm_bias: jax.Array,
    rope_cos: jax.Array,
    rope_sin: jax.Array,
    hadamard: jax.Array,
    index_key_cache: jax.Array,
    target_pages: jax.Array,
    *,
    page_size: int = 64,
    interpret: bool = False,
    token_slots: bool = False,
) -> jax.Array:
    """Project into whole pages, or preserve partial pages with token_slots."""

    if page_size != 64:
        raise ValueError("only 64-token pages are supported")
    if hidden_states.dtype != jnp.bfloat16 or wk.dtype != jnp.float8_e4m3fn:
        raise ValueError("expected BF16 activations and E4M3FN weights")
    if scale.shape != (128,) or scale.dtype != jnp.float32:
        raise ValueError("expected per-output-channel float32 scale[128]")
    if index_key_cache.dtype != jnp.bfloat16:
        raise ValueError("expected BF16 cache")
    if hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must be rank 2, got {hidden_states.shape}")
    if token_slots:
        tokens = hidden_states.shape[0]
        if target_pages.shape != (tokens,) or target_pages.dtype != jnp.int32:
            raise ValueError("token_slots requires one int32 physical slot per token")
        padding = (-tokens) % page_size
        hidden_states = jnp.pad(hidden_states, ((0, padding), (0, 0)))
        rope_cos = jnp.pad(rope_cos, ((0, padding), (0, 0)))
        rope_sin = jnp.pad(rope_sin, ((0, padding), (0, 0)))
        target_pages = jnp.pad(target_pages, (0, padding), constant_values=-1)
    num_tokens, hidden_dim = hidden_states.shape
    if num_tokens < page_size or num_tokens % page_size:
        raise ValueError(
            f"num_tokens={num_tokens} must be a positive multiple of page_size={page_size}"
        )
    if index_key_cache.ndim != 3 or index_key_cache.shape[1:] != (page_size, 128):
        raise ValueError(
            "index_key_cache must have shape [num_pages, page_size, 128], "
            f"got {index_key_cache.shape}"
        )
    if wk.shape != (128, hidden_dim):
        raise ValueError(f"wk must have shape {(128, hidden_dim)}, got {wk.shape}")
    if norm_weight.shape != (128,) or norm_bias.shape != (128,):
        raise ValueError("index-key norm parameters must both have shape [128]")
    if rope_cos.shape != (num_tokens, 32) or rope_sin.shape != (num_tokens, 32):
        raise ValueError("rope_cos and rope_sin must both have shape [T, 32]")
    if hadamard.shape != (128, 128) or hadamard.dtype != jnp.float32:
        raise ValueError("hadamard must be float32[128, 128]")

    rope_cos = rope_cos.astype(jnp.bfloat16)
    rope_sin = rope_sin.astype(jnp.bfloat16)
    num_token_pages = num_tokens // page_size
    expected_targets = num_tokens if token_slots else num_token_pages
    if target_pages.shape != (expected_targets,) or target_pages.dtype != jnp.int32:
        raise ValueError(
            f"target_pages must be int32[{num_token_pages}], got {target_pages.shape} "
            f"and {target_pages.dtype}"
        )

    inputs = (
        hidden_states,
        wk,
        scale.reshape(1, 128),
        norm_weight,
        norm_bias,
        rope_cos,
        rope_sin,
        hadamard,
        index_key_cache,
    )
    in_specs = (
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(wk.shape, lambda *_: (0, 0)),
        pl.BlockSpec((1, 128), lambda *_: (0, 0)),
        pl.BlockSpec(norm_weight.shape, lambda *_: (0,)),
        pl.BlockSpec(norm_bias.shape, lambda *_: (0,)),
        pl.BlockSpec(rope_cos.shape, lambda *_: (0, 0)),
        pl.BlockSpec(rope_sin.shape, lambda *_: (0, 0)),
        pl.BlockSpec(hadamard.shape, lambda *_: (0, 0)),
        pl.BlockSpec(memory_space=pltpu.HBM),
    )
    cache_input_index = len(inputs)
    return pl.pallas_call(
        functools.partial(
            _fused_fp8_k_projection_scatter_kernel,
            token_slots=token_slots,
            num_token_pages=num_token_pages,
            page_size=page_size,
        ),
        out_shape=jax.ShapeDtypeStruct(index_key_cache.shape, index_key_cache.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=in_specs,
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.VMEM((2, page_size, hidden_dim), hidden_states.dtype),
                pltpu.VMEM((2, page_size, 128), index_key_cache.dtype),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.VMEM((page_size, 128), index_key_cache.dtype),
                pltpu.SemaphoreType.DMA,
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=128 * 1024 * 1024,
            disable_bounds_checks=True,
        ),
        input_output_aliases={cache_input_index: 0},
        interpret=interpret,
        name="dsa_fused_fp8_k_projection_scatter_aligned",
    )(target_pages, *inputs)


__all__ = ["fused_fp8_k_projection_scatter_pallas"]
