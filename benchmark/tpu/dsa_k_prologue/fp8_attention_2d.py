# SPDX-License-Identifier: Apache-2.0
# TEST ONLY (experiment B): copy of source_dsa/kernels/dsa/fp8_attention.py with one change:
# the gathered latent tile is passed and staged as [.., K*4, 128] instead of [.., K, 4, 128],
# so Mosaic can tile it (32,128) like Target. Everything else is unchanged.
# Original header: Verbatim copy of primatrix-glm/sglang-jax-private (Apache-2.0) at c948bdda95208e68a474c8088937347d08fe8937:
#   python/sgl_jax/srt/kernels/dsa/fp8_attention.py
# Only change: import prefix 'sgl_jax.srt.' -> 'sglang_tpu.attention.pallas.source_dsa.'. See source_dsa/MANIFEST.json.
"""FP8 QK/PV attention over packed per-row-scaled DSA KV cache.

Both exact-DSA schedulers share this TensorCore implementation. Queries enter
as BF16 and are quantized once per tile in VMEM. Softmax statistics and output
accumulation stay FP32. The default output is BF16; partial Attention exports
the normalized FP32 accumulator and natural-log LSE before any BF16 cast.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.fp8_kv_cache import (
    FP8_KV_CACHE_SCALE_BYTES,
    GatheredFP8KVGroups,
    SplitFP8KVRows,
    cache_nbytes,
    cache_parts,
    cache_primary,
    flatten_split_fp8_tile,
    map_cache_parts,
)
from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.partial_attention import normalize_attention_output


def validate_fp8_attention_config(fp8_dtype, bq: int):
    """Reject unvalidated FP8 formats/VMEM tile sizes before TPU lowering."""
    fp8_dtype = fp8_dtype or jnp.float8_e4m3fn
    if jnp.dtype(fp8_dtype) != jnp.dtype(jnp.float8_e4m3fn):
        raise TypeError(
            "FP8 Q attention supports only float8_e4m3fn KV cache; "
            "use --dsa-attention-precision bf16 for E5M2 cache."
        )
    if not 0 < bq <= 16:
        raise ValueError(
            "FP8 Q attention requires 1 <= bq <= 16 for the supported VMEM configuration; "
            "use --dsa-extend-tiles 256,16,256 or 256,16,512."
        )
    return fp8_dtype


def _align_to(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _unpack_scale(cache, offset):
    """Recover exact FP32 scale bits with one transpose per scale tile."""
    scale_bytes = cache[..., offset : offset + FP8_KV_CACHE_SCALE_BYTES]
    return pltpu.bitcast(jnp.swapaxes(scale_bytes, -2, -1), jnp.float32).reshape(cache.shape[:-1])


def _dsa_tensor_core_kernel(
    sm_scale_ref,  # FP32 scalar, SMEM
    selected_counts_ref,  # INT32 [B], SMEM
    cache_buffer_index_ref,  # INT32 scalar, SMEM
    q_hbm_ref,  # [B, H, align(V + R, 128)], HBM
    cache_hbm_ref,  # [pool_slots, microbatch_queries, K, 4, D], HBM; split path uses 1 slot
    output_hbm_ref,  # [B, H, V_pad], HBM
    q_x2_ref,  # [2, bq, H, align(V + R, 128)], VMEM
    cache_x2_ref,  # [2, bq, b_topk, C_pad], VMEM
    output_x2_ref,  # [2, bq, H, V_pad], VMEM
    dma_sems,  # DMA semaphore [3, 2]: q, cache, output
    m_ref,  # FP32 [bq, H, 128], VMEM online-softmax maxima
    l_ref,  # FP32 [bq, H, 128], VMEM online-softmax denominators
    accumulator_ref,  # FP32 [bq, H, V_pad], VMEM value accumulator
    q8_ref,  # FP8 [bq, H, Q_pad], VMEM
    q_scale_ref,  # FP32 [bq, H, 128], VMEM
    *,
    num_q_steps: int,
    num_k_steps: int,
    bq: int,
    b_topk: int,
    latent_dim: int,
    rope_dim: int,
    padded_latent_dim: int,
    padded_q_dim: int,
    fp8_dtype,
    fp8_pv: bool,
    return_lse: bool = False,
):
    """Single-program online-softmax kernel with explicit HBM/VMEM DMA."""

    def async_copy(src, dst, sem, *, wait: bool):
        copy = pltpu.make_async_copy(src, dst, sem)
        if wait:
            copy.wait()
        else:
            copy.start()

    @jax.named_scope("query_dma")
    def fetch_q(q_step, buffer, *, wait: bool):
        q_start = q_step * bq
        q_dst = q_x2_ref.at[buffer]
        sem = dma_sems.at[0, buffer]
        if wait:
            async_copy(q_dst, q_dst, sem, wait=True)
        else:
            async_copy(
                q_hbm_ref.at[pl.ds(q_start, bq)],
                q_dst,
                sem,
                wait=False,
            )

    @jax.named_scope("cache_dma")
    def fetch_cache(q_step, k_step, buffer, *, wait: bool):
        for part_id, (cache_source, cache_bank) in enumerate(zip(
            cache_parts(cache_hbm_ref), cache_parts(cache_x2_ref), strict=True
        )):
            rows = 4 if part_id == 0 else 1  # latent part: 4 rows of 128 B per key
            cache_dst = cache_bank.at[buffer]
            sem = dma_sems.at[1, buffer]
            if wait:
                async_copy(cache_dst, cache_dst, sem, wait=True)
            else:
                async_copy(
                    cache_source.at[
                        cache_buffer_index_ref[()],
                        pl.ds(q_step * bq, bq),
                        pl.ds(k_step * b_topk * rows, b_topk * rows),
                    ],
                    cache_dst,
                    sem,
                    wait=False,
                )

    @jax.named_scope("output_dma")
    def send_output(q_step, buffer, *, wait: bool):
        output_src = output_x2_ref.at[buffer]
        sem = dma_sems.at[2, buffer]
        output_dst = output_hbm_ref.at[pl.ds(q_step * bq, bq)]
        async_copy(output_src, output_dst, sem, wait=wait)

    # Prologue: seed the first query and cache buffers.
    fetch_q(0, 0, wait=False)
    fetch_cache(0, 0, 0, wait=False)

    def compute_q(q_step, _):
        q_buffer = q_step % 2
        next_q_buffer = 1 - q_buffer

        @pl.when(q_step + 1 < num_q_steps)
        def prefetch_next_q():
            fetch_q(q_step + 1, next_q_buffer, wait=False)

        fetch_q(q_step, q_buffer, wait=True)
        # Quantize once per query tile, outside the K loop. The same scale
        # covers latent and rotary channels; padding is zero before reduction.
        with jax.named_scope("fp8_query_quantize"):
            query = q_x2_ref[q_buffer].astype(jnp.float32)
            q_scale = jnp.max(jnp.abs(query), axis=-1, keepdims=True) / float(
                jnp.finfo(fp8_dtype).max
            )
            q_scale = jnp.where(q_scale > 0, q_scale, 1.0)
            q8_ref[...] = jnp.clip(
                query / q_scale, -float(jnp.finfo(fp8_dtype).max), float(jnp.finfo(fp8_dtype).max)
            ).astype(fp8_dtype)
            q_scale_ref[...] = jnp.broadcast_to(q_scale, q_scale_ref.shape)
        m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
        l_ref[...] = jnp.zeros_like(l_ref)
        accumulator_ref[...] = jnp.zeros_like(accumulator_ref)

        active_k_steps = num_k_steps
        if return_lse:
            # Scalar SMEM loads avoid a VMEM-to-scalar reduction.
            max_count = jnp.int32(0)
            for i in range(bq):
                max_count = jnp.maximum(max_count, selected_counts_ref[q_step * bq + i])
            max_count = jnp.minimum(max_count, num_k_steps * b_topk)
            active_k_steps = (max_count + b_topk - 1) // b_topk

        def compute_k(k_step, _):
            cache_buffer = k_step % 2
            next_cache_buffer = 1 - cache_buffer

            @pl.when(k_step + 1 < active_k_steps)
            def prefetch_next_cache():
                fetch_cache(q_step, k_step + 1, next_cache_buffer, wait=False)

            fetch_cache(q_step, k_step, cache_buffer, wait=True)
            q = q8_ref[...]
            # SMEM supports scalar loads only. Static unrolling constructs the VMEM
            # vector without asking Mosaic to lower a vector-valued SMEM load.
            selected_counts = jnp.stack([selected_counts_ref[q_step * bq + i] for i in range(bq)])
            if isinstance(cache_x2_ref, (SplitFP8KVRows, GatheredFP8KVGroups)):
                tile = map_cache_parts(lambda ref: ref[cache_buffer], cache_x2_ref)
                latent = tile.latent.reshape(bq, b_topk, -1)[..., :latent_dim]
                rope = tile.rope.reshape(bq, b_topk, tile.rope.shape[-1])
                cache = jnp.concatenate((latent, rope), axis=-1)
            else:
                cache = cache_x2_ref[cache_buffer].reshape(bq, b_topk, -1)
            # Unpack without dequantizing. Never interpret the packed scale
            # bytes as QK padding: bytes [latent+rope:Q_pad] must become zeros.
            quantized_dim = latent_dim + rope_dim
            quantized = pltpu.bitcast(cache[..., :quantized_dim], fp8_dtype)
            # Transpose the whole scale tile [bq, K, 4] -> [bq, 4, K]
            # before bitcasting. Keeping K as a vector axis avoids the old
            # per-row [1,4] transpose (thousands of vxpose instructions).
            scale = _unpack_scale(cache, quantized_dim)
            cache = jnp.pad(quantized, ((0, 0), (0, 0), (0, padded_q_dim - quantized_dim)))

            with jax.named_scope("qk"):
                scores = lax.dot_general(
                    q,
                    cache[..., :padded_q_dim],
                    dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                    preferred_element_type=jnp.float32,
                )
                scores *= q_scale_ref[..., :1] * scale[:, None, :] * sm_scale_ref[()]
            with jax.named_scope("softmax"):
                k_offsets = k_step * b_topk + jnp.arange(b_topk, dtype=jnp.int32)
                valid = k_offsets[None, None, :] < selected_counts[:, None, None]
                masked_scores = jnp.where(valid, scores, jnp.float32(-1.0e30))

                old_m = m_ref[..., 0]
                old_l = l_ref[..., 0]
                tile_m = jnp.max(masked_scores, axis=-1)
                new_m = jnp.maximum(old_m, tile_m)
                alpha = jnp.exp(old_m - new_m)
                probabilities = jnp.exp(masked_scores - new_m[..., None]) * valid
                new_l = old_l * alpha + jnp.sum(probabilities, axis=-1)
            # KV scales vary along the PV contraction axis. Absorb them
            # into the probabilities, not the final output or denominator.
            if fp8_pv:
                with jax.named_scope("fp8_probability_quantize"):
                    weighted = probabilities * scale[:, None, :]
                    p_scale = jnp.max(weighted, axis=-1, keepdims=True) / float(
                        jnp.finfo(fp8_dtype).max
                    )
                    p_scale = jnp.where(p_scale > 0, p_scale, 1.0)
                    p8 = jnp.clip(weighted / p_scale, 0.0, float(jnp.finfo(fp8_dtype).max)).astype(
                        fp8_dtype
                    )
                with jax.named_scope("pv"):
                    value_update = (
                        lax.dot_general(
                            p8,
                            cache[..., :padded_latent_dim],
                            dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                            preferred_element_type=jnp.float32,
                        )
                        * p_scale
                    )
            else:
                # Precision ablation: retain FP8 QK and packed HBM/gather,
                # but dequantize V in VMEM and avoid quantizing probabilities.
                with jax.named_scope("pv_bf16"):
                    values = (
                        cache[..., :padded_latent_dim].astype(jnp.float32) * scale[..., None]
                    ).astype(jnp.bfloat16)
                    value_update = lax.dot_general(
                        probabilities.astype(jnp.bfloat16),
                        values,
                        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                        preferred_element_type=jnp.float32,
                    )
            accumulator_ref[...] = accumulator_ref[...] * alpha[..., None] + value_update
            m_ref[...] = jnp.broadcast_to(new_m[..., None], m_ref.shape)
            l_ref[...] = jnp.broadcast_to(new_l[..., None], l_ref.shape)
            return None

        if return_lse:
            # Empty tiles still own the prologue/previous query's seed DMA.
            # Drain it before the next query reuses cache bank zero.
            @pl.when(active_k_steps == 0)
            def drain_empty_cache():
                fetch_cache(q_step, 0, 0, wait=True)

        lax.fori_loop(0, active_k_steps, compute_k, None, unroll=False)

        output_buffer = q_step % 2

        @pl.when(q_step >= 2)
        def wait_old_output():
            send_output(q_step - 2, output_buffer, wait=True)

        output_x2_ref[output_buffer] = normalize_attention_output(
            accumulator_ref[...], m_ref[..., 0], l_ref[..., 0], return_lse=return_lse
        ).astype(output_x2_ref.dtype)
        send_output(q_step, output_buffer, wait=False)

        @pl.when(q_step + 1 < num_q_steps)
        def seed_next_query_cache():
            fetch_cache(q_step + 1, 0, 0, wait=False)

        return None

    lax.fori_loop(0, num_q_steps, compute_q, None, unroll=False)

    # Epilogue: drain the at most two outstanding output transfers.
    for q_step in range(max(0, num_q_steps - 2), num_q_steps):
        send_output(q_step, q_step % 2, wait=True)


def fp8_tensor_core_attention(
    q_latent: jax.Array,
    q_rope: jax.Array,
    gathered_cache: jax.Array,
    selected_counts: jax.Array,
    cache_buffer_index: jax.Array,
    *,
    sm_scale: float,
    bq: int,
    b_topk: int,
    fp8_dtype=None,
    fp8_pv: bool = False,
    return_lse: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Run the TensorCore online-softmax stage for one SparseCore microbatch."""
    batch_size, num_heads, _ = q_latent.shape
    latent_dim = q_latent.shape[-1]
    rope_dim = q_rope.shape[-1]
    q_dim = latent_dim + rope_dim
    topk = cache_primary(gathered_cache).shape[2]
    lat = gathered_cache.latent
    gathered_cache = SplitFP8KVRows(
        lat.reshape(*lat.shape[:2], topk * lat.shape[3], lat.shape[4]), gathered_cache.rope
    )
    if cache_primary(gathered_cache).dtype != jnp.uint8:
        raise TypeError("FP8 Q/PV requires packed uint8 KV cache.")
    fp8_dtype = validate_fp8_attention_config(fp8_dtype, bq)
    num_q_steps = batch_size // bq
    num_k_steps = topk // b_topk
    padded_latent_dim = _align_to(latent_dim, 128)
    output_dim = padded_latent_dim + (128 if return_lse else 0)
    output_dtype = jnp.float32 if return_lse else q_latent.dtype
    padded_q_dim = _align_to(q_dim, 128)
    q = jnp.concatenate(
        (
            q_latent,
            q_rope,
            jnp.zeros(
                (*q_latent.shape[:-1], padded_q_dim - q_dim),
                dtype=q_latent.dtype,
            ),
        ),
        axis=-1,
    )

    padded_output = pl.pallas_call(
        functools.partial(
            _dsa_tensor_core_kernel,
            num_q_steps=num_q_steps,
            num_k_steps=num_k_steps,
            bq=bq,
            b_topk=b_topk,
            latent_dim=latent_dim,
            rope_dim=rope_dim,
            padded_latent_dim=padded_latent_dim,
            padded_q_dim=padded_q_dim,
            fp8_dtype=fp8_dtype,
            fp8_pv=fp8_pv,
            return_lse=return_lse,
        ),
        out_shape=jax.ShapeDtypeStruct(
            (batch_size, num_heads, output_dim),
            output_dtype,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                map_cache_parts(lambda _: pl.BlockSpec(memory_space=pltpu.HBM), gathered_cache),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((2, bq, num_heads, padded_q_dim), q_latent.dtype),
                map_cache_parts(
                    lambda part: pltpu.VMEM(
                        (2, bq, b_topk * (part.shape[2] // topk), *part.shape[3:]), part.dtype
                    ),
                    gathered_cache,
                ),
                pltpu.VMEM(
                    (2, bq, num_heads, output_dim),
                    output_dtype,
                ),
                pltpu.SemaphoreType.DMA((3, 2)),
                pltpu.VMEM((bq, num_heads, 128), jnp.float32),
                pltpu.VMEM((bq, num_heads, 128), jnp.float32),
                pltpu.VMEM((bq, num_heads, padded_latent_dim), jnp.float32),
                pltpu.VMEM((bq, num_heads, padded_q_dim), fp8_dtype),
                pltpu.VMEM((bq, num_heads, 128), jnp.float32),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(),
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
        cost_estimate=pl.CostEstimate(
            flops=2 * batch_size * num_heads * topk * (padded_q_dim + padded_latent_dim),
            bytes_accessed=q.size * q.dtype.itemsize
            # Query tiles read disjoint rows of one active cache microbatch.
            + cache_nbytes(gathered_cache) // cache_primary(gathered_cache).shape[0]
            + batch_size * num_heads * output_dim * jnp.dtype(output_dtype).itemsize
            + selected_counts.size * selected_counts.dtype.itemsize
            + jnp.dtype(jnp.float32).itemsize  # sm_scale
            + jnp.dtype(jnp.int32).itemsize,  # cache_buffer_index
            transcendentals=batch_size * num_heads * (num_k_steps + topk),
        ),
        name=(
            "dsa_tensor_core_attention_fp8_qkv_2d"
            if fp8_pv
            else "dsa_tensor_core_attention_fp8_q_bf16_pv"
        )
        + ("_partial" if return_lse else ""),
    )(
        jnp.asarray(sm_scale, dtype=jnp.float32),
        selected_counts,
        cache_buffer_index,
        q,
        gathered_cache,
    )
    if return_lse:
        return padded_output[..., :latent_dim], padded_output[..., padded_latent_dim]
    return padded_output[..., :latent_dim]
