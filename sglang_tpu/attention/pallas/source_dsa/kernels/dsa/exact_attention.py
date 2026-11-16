# SPDX-License-Identifier: Apache-2.0
# Verbatim copy of primatrix-glm/sglang-jax-private (Apache-2.0) at c948bdda95208e68a474c8088937347d08fe8937:
#   python/sgl_jax/srt/kernels/dsa/exact_attention.py
# Only change: import prefix 'sgl_jax.srt.' -> 'sglang_tpu.attention.pallas.source_dsa.'. See source_dsa/MANIFEST.json.
"""TPU SparseCore + TensorCore implementation of exact DSA attention.

Adapted from ``primatrix/pallas-kernel`` commit ``63898542``
(``tops/ops/dsa/dsa.py``). The only source-level adaptation removes the
optional jaxtyping dependency so this kernel follows sgl-jax's array API.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.attention_inputs import validate_attention_inputs
from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.fp8_kv_cache import (
    FP8_KV_CACHE_SCALE_BYTES,
    GatheredFP8KVGroups,
    SplitFP8KVCache,
    SplitFP8KVRows,
    cache_nbytes,
    cache_parts,
    cache_primary,
    flatten_split_fp8_tile,
    map_cache_parts,
)
from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.gather import (
    _SC_GATHER_WINDOW,
    _align_to,
    _gather_split_fp8,
    _gather_split_fp8_groups,
    _sparse_core_gather,
)
from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.gather import (
    _sparse_core_gather_cost_estimate as _sparse_core_gather_cost_estimate,
)
from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.gather import _sparse_core_info
from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.partial_attention import normalize_attention_output


def _gather_cache_microbatch_into(
    cache: jax.Array,
    safe_slots: jax.Array,
    output_pool: jax.Array,
    output_buffer_index: jax.Array | int,
) -> jax.Array:
    """Gather one cache microbatch into one slot of ``output_pool``."""
    return _sparse_core_gather(
        cache,
        safe_slots.reshape(-1),
        output_buffer=output_pool,
        output_buffer_index=output_buffer_index,
        skip_invalid=True,
    )


@jax.named_scope("dsa_tc")
def _dsa_tensor_core_kernel(
    sm_scale_ref,  # FP32 scalar, SMEM
    selected_counts_ref,  # INT32 [B], SMEM
    cache_buffer_index_ref,  # INT32 scalar, SMEM
    q_hbm_ref,  # [B, H, align(V + R, 128)], HBM
    cache_hbm_ref,  # [2, B, K, pair_dim, D], HBM; first dim is the ping-pong pool
    output_hbm_ref,  # [B, H, V_pad], HBM
    q_x2_ref,  # [2, bq, H, align(V + R, 128)], VMEM
    cache_x2_ref,  # [2, bq, b_topk, C_pad], VMEM
    output_x2_ref,  # [2, bq, H, V_pad], VMEM
    dma_sems,  # DMA semaphore [3, 2]: q, cache, output
    m_ref,  # FP32 [bq, H, 128], VMEM online-softmax maxima
    l_ref,  # FP32 [bq, H, 128], VMEM online-softmax denominators
    accumulator_ref,  # FP32 [bq, H, V_pad], VMEM value accumulator
    decoded_cache_x2_ref=None,  # FP8 path only: BF16 [2, bq, b_topk, Q_pad]
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
    return_lse: bool = False,
):
    """Single-program online-softmax kernel with explicit HBM/VMEM DMA.

    Named scopes label TPU device trace regions without adding explicit DMA
    waits or compute barriers. DMA start scopes measure submission, while wait
    scopes expose remaining stalls; overlapping stage durations are not additive.
    """

    def async_copy(src, dst, sem, *, wait: bool):
        with jax.named_scope("wait" if wait else "start"):
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
        for cache_source, cache_bank in zip(
            cache_parts(cache_hbm_ref), cache_parts(cache_x2_ref), strict=True
        ):
            cache_dst = cache_bank.at[buffer]
            sem = dma_sems.at[1, buffer]
            if wait:
                async_copy(cache_dst, cache_dst, sem, wait=True)
            else:
                async_copy(
                    cache_source.at[
                        cache_buffer_index_ref[()],
                        pl.ds(q_step * bq, bq),
                        pl.ds(k_step * b_topk, b_topk),
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

    @jax.named_scope("fp8_decode")
    def decode_cache(buffer):
        if isinstance(cache_x2_ref, (SplitFP8KVRows, GatheredFP8KVGroups)):
            cache = flatten_split_fp8_tile(
                map_cache_parts(lambda ref: ref[buffer], cache_x2_ref), latent_dim
            )
        else:
            cache = cache_x2_ref[buffer].reshape(bq, b_topk, -1)
        quantized_dim = latent_dim + rope_dim
        quantized = pltpu.bitcast(cache[..., :quantized_dim], fp8_dtype).astype(jnp.float32)
        # The whole latent+RoPE row shares one FP32 scale. Mosaic packs
        # the four scale bytes along the second-minor axis for bitcast.
        scale_bytes = cache[..., quantized_dim : quantized_dim + FP8_KV_CACHE_SCALE_BYTES].reshape(
            bq, b_topk, 1, FP8_KV_CACHE_SCALE_BYTES
        )
        scale = pltpu.bitcast(jnp.swapaxes(scale_bytes, -2, -1), jnp.float32).reshape(bq, b_topk, 1)
        values = (quantized * scale).astype(jnp.bfloat16)
        if padded_q_dim != quantized_dim:
            values = jnp.pad(values, ((0, 0), (0, 0), (0, padded_q_dim - quantized_dim)))
        decoded_cache_x2_ref[buffer] = values

    # Prologue: seed the first query and cache buffers.
    fetch_q(0, 0, wait=False)
    fetch_cache(0, 0, 0, wait=False)

    def compute_q(q_step, _):
        q_buffer = q_step % 2
        next_q_buffer = 1 - q_buffer
        active_k_steps = num_k_steps
        if return_lse:
            max_count = selected_counts_ref[q_step * bq]
            for i in range(1, bq):
                max_count = jnp.maximum(max_count, selected_counts_ref[q_step * bq + i])
            # Always consume the seeded DMA, even for an empty partition.
            # Masking that first tile still produces exactly zero output/-inf LSE.
            active_k_steps = jnp.clip((max_count + b_topk - 1) // b_topk, 1, num_k_steps)

        @pl.when(q_step + 1 < num_q_steps)
        def prefetch_next_q():
            fetch_q(q_step + 1, next_q_buffer, wait=False)

        fetch_q(q_step, q_buffer, wait=True)
        with jax.named_scope("init_state"):
            m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
            l_ref[...] = jnp.zeros_like(l_ref)
            accumulator_ref[...] = jnp.zeros_like(accumulator_ref)

        if fp8_dtype is not None:
            with jax.named_scope("prologue"):
                # The first raw tile is seeded by the kernel/query prologue.
                # Start tile 1 before decoding tile 0 to give its DMA a head start.
                if num_k_steps > 1:

                    @pl.when(active_k_steps > 1)
                    def seed_second_cache():
                        fetch_cache(q_step, 1, 1, wait=False)

                fetch_cache(q_step, 0, 0, wait=True)
                decode_cache(0)
                # QK/PV consume the separate decoded bank, so decoding tile 0
                # releases its raw bank for tile 2 before attention starts.
                if num_k_steps > 2:

                    @pl.when(active_k_steps > 2)
                    def seed_third_cache():
                        fetch_cache(q_step, 2, 0, wait=False)

        def compute_k(k_step, _, *, decode_next=False):
            cache_buffer = k_step % 2
            next_cache_buffer = 1 - cache_buffer

            if fp8_dtype is None:

                @pl.when(k_step + 1 < active_k_steps)
                def prefetch_next_cache():
                    fetch_cache(q_step, k_step + 1, next_cache_buffer, wait=False)

                fetch_cache(q_step, k_step, cache_buffer, wait=True)
            q = q_x2_ref[q_buffer]
            # SMEM supports scalar loads only. Static unrolling constructs the VMEM
            # vector without asking Mosaic to lower a vector-valued SMEM load.
            selected_counts = jnp.stack([selected_counts_ref[q_step * bq + i] for i in range(bq)])
            if fp8_dtype is not None:
                cache = decoded_cache_x2_ref[cache_buffer]
            else:
                cache = cache_x2_ref[cache_buffer].reshape(
                    (bq, b_topk, cache_x2_ref.shape[-2] * cache_x2_ref.shape[-1])
                )

            with jax.named_scope("qk"):
                scores = lax.dot_general(
                    q,
                    cache[..., :padded_q_dim],
                    dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                    preferred_element_type=jnp.float32,
                )
            with jax.named_scope("softmax"):
                scores *= sm_scale_ref[()]

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
            if decode_next:
                # Complete the next tile's DMA before PV so its wait is outside
                # the current-PV / next-tile-decode scheduling window.
                fetch_cache(q_step, k_step + 1, next_cache_buffer, wait=True)
            with jax.named_scope("pv"):
                value_update = lax.dot_general(
                    probabilities.astype(q.dtype),
                    cache[..., :padded_latent_dim],
                    dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                    preferred_element_type=jnp.float32,
                )

            if decode_next:
                # Independent of value_update: expose next-tile VPU decode
                # alongside the current PV to the TPU instruction scheduler.
                decode_cache(next_cache_buffer)

                # Refill the raw bank just released by decode(k+1).
                # Tile k+2 is already in flight from the preceding decode;
                # prefetch k+3 without changing the QK/softmax/PV order.
                @pl.when(k_step + 3 < active_k_steps)
                def prefetch_future_cache():
                    fetch_cache(q_step, k_step + 3, next_cache_buffer, wait=False)

            with jax.named_scope("accumulate"):
                accumulator_ref[...] = accumulator_ref[...] * alpha[..., None] + value_update
                m_ref[...] = jnp.broadcast_to(new_m[..., None], m_ref.shape)
                l_ref[...] = jnp.broadcast_to(new_l[..., None], l_ref.shape)
            return None

        if fp8_dtype is not None:
            if num_k_steps > 1:
                with jax.named_scope("steady_state"):
                    lax.fori_loop(
                        0,
                        active_k_steps - 1,
                        functools.partial(compute_k, decode_next=True),
                        None,
                        unroll=False,
                    )
            # Drain the last decoded tile without waiting for a nonexistent
            # next DMA. This is also the complete single-tile path.
            with jax.named_scope("drain"):
                compute_k(active_k_steps - 1, None)
        else:
            with jax.named_scope("steady_state"):
                lax.fori_loop(0, active_k_steps, compute_k, None, unroll=False)

        output_buffer = q_step % 2

        @pl.when(q_step >= 2)
        def wait_old_output():
            send_output(q_step - 2, output_buffer, wait=True)

        with jax.named_scope("output_normalize"):
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


def _tensor_core_attention(
    q_latent: jax.Array,
    q_rope: jax.Array,
    cache_buffers: jax.Array,
    selected_counts: jax.Array,
    cache_buffer_index: jax.Array,
    *,
    sm_scale: float,
    bq: int,
    b_topk: int,
    fp8_dtype=None,
    fp8_q: bool = False,
    fp8_pv: bool = False,
    return_lse: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Run the TensorCore online-softmax stage for one SparseCore microbatch."""
    if fp8_q:
        from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.fp8_attention import fp8_tensor_core_attention

        return fp8_tensor_core_attention(
            q_latent,
            q_rope,
            cache_buffers,
            selected_counts,
            cache_buffer_index,
            sm_scale=sm_scale,
            bq=bq,
            b_topk=b_topk,
            fp8_dtype=fp8_dtype,
            fp8_pv=fp8_pv,
            return_lse=return_lse,
        )
    batch_size, num_heads, _ = q_latent.shape
    latent_dim = q_latent.shape[-1]
    rope_dim = q_rope.shape[-1]
    q_dim = latent_dim + rope_dim
    topk = cache_primary(cache_buffers).shape[2]
    num_q_steps = batch_size // bq
    num_k_steps = topk // b_topk
    cache_is_fp8 = cache_primary(cache_buffers).dtype == jnp.uint8
    fp8_dtype = (fp8_dtype or jnp.float8_e4m3fn) if cache_is_fp8 else None
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

    # Query tiles stream disjoint query rows of one cache microbatch. Across
    # the Q/K loops each active byte is read once, not num_q_steps times.
    # Count padded dimensions, as consumed by the dot operations and DMA.
    flops = 2 * batch_size * num_heads * topk * (padded_q_dim + padded_latent_dim)
    transcendentals = batch_size * num_heads * (num_k_steps + topk)
    bytes_accessed = (
        q.size * q.dtype.itemsize
        + cache_nbytes(cache_buffers) // cache_primary(cache_buffers).shape[0]
        + batch_size * num_heads * output_dim * jnp.dtype(output_dtype).itemsize
        + jnp.dtype(jnp.float32).itemsize  # sm_scale
        + jnp.dtype(jnp.int32).itemsize  # cache_buffer_index
        + selected_counts.size * selected_counts.dtype.itemsize
    )
    cost_estimate = pl.CostEstimate(
        flops=flops,
        bytes_accessed=bytes_accessed,
        transcendentals=transcendentals,
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
                map_cache_parts(lambda _: pl.BlockSpec(memory_space=pltpu.HBM), cache_buffers),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((2, bq, num_heads, padded_q_dim), q_latent.dtype),
                map_cache_parts(
                    lambda part: pltpu.VMEM((2, bq, b_topk, *part.shape[3:]), part.dtype),
                    cache_buffers,
                ),
                pltpu.VMEM(
                    (2, bq, num_heads, output_dim),
                    output_dtype,
                ),
                pltpu.SemaphoreType.DMA((3, 2)),
                pltpu.VMEM((bq, num_heads, 128), jnp.float32),
                pltpu.VMEM((bq, num_heads, 128), jnp.float32),
                pltpu.VMEM((bq, num_heads, padded_latent_dim), jnp.float32),
                *(
                    [pltpu.VMEM((2, bq, b_topk, padded_q_dim), jnp.bfloat16)]
                    if cache_is_fp8
                    else []
                ),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(),
            vmem_limit_bytes=64 * 1024 * 1024,
        ),
        cost_estimate=cost_estimate,
        name="dsa_tensor_core_attention" + ("_partial" if return_lse else ""),
    )(
        jnp.asarray(sm_scale, dtype=jnp.float32),
        selected_counts,
        cache_buffer_index,
        q,
        cache_buffers,
    )
    if return_lse:
        return padded_output[..., :latent_dim], padded_output[..., padded_latent_dim]
    return padded_output[..., :latent_dim]


@functools.partial(
    jax.jit,
    static_argnames=(
        "bq_sparse",
        "bq",
        "b_topk",
        "fp8_dtype",
        "fp8_q",
        "fp8_pv",
        "split_gather_buffers",
        "gather_rope_groups",
        "return_lse",
    ),
)
def sparse_core_tensor_core_dsa(
    q_latent: jax.Array,
    q_rope: jax.Array,
    cache: jax.Array,
    physical_slots: jax.Array,
    selected_counts: jax.Array,
    sm_scale: jax.Array | float,
    *,
    bq_sparse: int = 128,
    bq: int = 16,
    b_topk: int = 128,
    fp8_dtype=None,
    fp8_q: bool = False,
    fp8_pv: bool = False,
    split_gather_buffers: bool = False,
    gather_rope_groups: bool = False,
    return_lse: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Compute sparse DSA attention with overlapped SparseCore and TensorCore work.

    Args:
      q_latent: BF16 latent queries, shape ``[Q, H, V]``. ``Q`` is the
        number of query tokens and ``H`` is the number of query heads.
      q_rope: Rotary query components, shape ``[Q, H, R]`` and the same dtype.
      cache: BF16 ``[S, 2, D]``, legacy fused FP8 ``[S, 4, D]``, or
        SplitFP8KVCache with latent ``[S, 4, 128]`` and rope ``[S//4, 4, 128]``
        for GLM-5.2. The FP8 parts share one scale stored in RoPE padding.
      physical_slots: INT32 cache-row indices in ``[0, S)``, shape ``[Q, K]``.
        Only entries before the corresponding ``selected_counts`` value are read.
        Remaining entries are replaced with -1, skip HBM reads, and gather zeros.
      selected_counts: INT32 valid top-k counts, shape ``[Q]``. Values are
        clamped to ``[0, K]``; a zero count produces an all-zero output row.
      sm_scale: FP32 scalar applied to attention logits.
      bq_sparse: Query microbatch consumed by each SparseCore gather.
      bq: TensorCore query tile. Must divide ``bq_sparse``.
      b_topk: TensorCore online-softmax tile. Must divide ``K``.
      fp8_q: Quantize Q in VMEM for native FP8 QK; fp8_pv controls PV.
        Requires uint8 FP8 KV storage; output stays BF16. Defaults to False.
      fp8_pv: Use FP8 probabilities and V when fp8_q is enabled. False keeps
        PV in BF16 as a precision ablation; QK and gathered storage stay FP8.
      gather_rope_groups: Experimental split-cache path that gathers four RoPE
        rows per key and selects the target row inside TensorCore. Increases
        gather output by 384 bytes plus one int32 selector per key.
      return_lse: Return normalized FP32 output and natural-log FP32 LSE for
        DCP. Empty partitions return zero output and negative-infinite LSE.

    Returns:
      Sparse-attention output with shape ``[Q, H, V]`` and query dtype.

    FP8 always uses separate gather buffers: up to 16 microbatches are
    statically unrolled with one-microbatch-ahead prefetch dependencies; larger
    buckets use a paired loop. For BF16 with split_gather_buffers=False,
    the outer ``lax.fori_loop`` carries one cache pool whose leading dimension
    is two. The low bit of ``batch_id`` selects the current slice and the
    opposite slice is used as the gather destination. The pool itself never
    changes position in the loop state. All launch parameters are static,
    making this API safe to call either directly or from another
    ``jax.jit``-compiled function.
    """
    if fp8_pv and not fp8_q:
        raise ValueError("FP8 PV requires FP8 QK.")
    if gather_rope_groups and not isinstance(cache, SplitFP8KVCache):
        raise TypeError("Grouped RoPE gather requires split FP8 storage.")
    if fp8_q and cache_primary(cache).dtype != jnp.uint8:
        raise TypeError("fp8_q requires packed uint8 FP8 KV cache.")
    if fp8_q:
        from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.fp8_attention import validate_fp8_attention_config

        fp8_dtype = validate_fp8_attention_config(fp8_dtype, bq)
        # FP8 always uses the bounded split-buffer pipeline, including direct
        # callers and serving without the HLO-overlap option or launch presets.
        split_gather_buffers = True
    num_queries, latent_dim, rope_dim, topk = validate_attention_inputs(
        q_latent,
        q_rope,
        cache,
        physical_slots,
        selected_counts,
        fp8_dtype=fp8_dtype,
        bq_sparse=bq_sparse,
        bq=bq,
        b_topk=b_topk,
    )
    if cache_primary(cache).dtype == jnp.uint8:
        fp8_dtype = fp8_dtype or jnp.float8_e4m3fn
    if isinstance(cache, SplitFP8KVCache):
        split_gather_buffers = True

    info = _sparse_core_info()
    gather_size = bq_sparse * topk
    gather_multiple = _SC_GATHER_WINDOW * info.num_cores * info.num_subcores
    if gather_size % gather_multiple:
        raise ValueError(
            f"bq_sparse * K must be divisible by {gather_multiple}, got {gather_size}."
        )

    num_microbatches = num_queries // bq_sparse
    clipped_counts = jnp.clip(selected_counts, 0, topk)
    valid_slots = jnp.arange(topk, dtype=jnp.int32)[None, :] < clipped_counts[:, None]
    safe_slots = jnp.where(valid_slots, physical_slots, jnp.int32(-1))
    safe_slot_batches = safe_slots.reshape(num_microbatches, bq_sparse, topk)
    count_batches = clipped_counts.reshape(num_microbatches, bq_sparse)
    q_latent_batches = q_latent.reshape(
        num_microbatches,
        bq_sparse,
        q_latent.shape[1],
        latent_dim,
    )
    q_rope_batches = q_rope.reshape(
        num_microbatches,
        bq_sparse,
        q_rope.shape[1],
        rope_dim,
    )

    def empty_output():
        if return_lse:
            return (
                jnp.empty(q_latent.shape, jnp.float32),
                jnp.empty(q_latent.shape[:-1], jnp.float32),
            )
        return jnp.empty_like(q_latent)

    def update_output(output, value, start):
        return jax.tree.map(
            lambda dest, src: lax.dynamic_update_slice_in_dim(dest, src, start, axis=0),
            output,
            value,
        )

    if split_gather_buffers:
        unroll_split_pipeline = (
            fp8_q or isinstance(cache, SplitFP8KVCache)
        ) and num_microbatches <= 16

        # Independent arrays avoid a data dependency from the next gather's
        # whole-pool update to attention reading the current slot. Actual
        # overlap and memory assignment still require a device trace.
        def gather_separate(batch_id, source_cache=None):
            if source_cache is None:
                source_cache = cache
            slots = lax.dynamic_index_in_dim(safe_slot_batches, batch_id, keepdims=False)
            if gather_rope_groups:
                gathered = _gather_split_fp8_groups(source_cache, slots.reshape(-1))
            elif isinstance(source_cache, SplitFP8KVCache):
                # Native measurements favor fusion for the unrolled path;
                # the paired loop still benefits from independent gathers.
                gathered = _gather_split_fp8(
                    source_cache, slots.reshape(-1), fused=return_lse and unroll_split_pipeline
                )
            else:
                gathered = map_cache_parts(
                    lambda part: _sparse_core_gather(
                        part,
                        slots.reshape(-1),
                        skip_invalid=True,
                        native_byte_gather=False,
                    ),
                    source_cache,
                )
            return map_cache_parts(
                lambda part: part.reshape(1, bq_sparse, topk, *part.shape[1:]), gathered
            )

        def attend(batch_id, current):
            return _tensor_core_attention(
                lax.dynamic_index_in_dim(q_latent_batches, batch_id, keepdims=False),
                lax.dynamic_index_in_dim(q_rope_batches, batch_id, keepdims=False),
                current,
                lax.dynamic_index_in_dim(count_batches, batch_id, keepdims=False),
                jnp.int32(0),
                sm_scale=sm_scale,
                bq=bq,
                b_topk=b_topk,
                fp8_dtype=fp8_dtype,
                fp8_q=fp8_q,
                fp8_pv=fp8_pv,
                return_lse=return_lse,
            )

        initial = gather_separate(0)

        def consume(batch_id, current, output):
            next_cache = gather_separate(batch_id + 1)
            value = attend(batch_id, current)
            output = update_output(output, value, batch_id * bq_sparse)
            return next_cache, output

        if unroll_split_pipeline:
            # Full-model buffer assignment inserts a gather-sized copy on
            # every while iteration, even with the paired loop below. Small
            # prefill buckets can avoid this carry entirely. Keep each launch
            # paired with its own consumer in the HLO scheduling pass.
            def append_output(output, value, start):
                # Materialize the larger FP32 partials once, after the pipeline.
                return (*output, value) if return_lse else update_output(output, value, start)

            current, output = initial, () if return_lse else empty_output()
            previous = None
            for batch_id in range(num_microbatches - 1):
                with jax.named_scope(f"fp8_pipeline_{batch_id}"):
                    source_cache = cache
                    if previous is not None:
                        # Bound prefetch to one microbatch ahead. Without
                        # this dependency XLA can launch every gather early
                        # and retain several GiB of gathered rows. All barrier
                        # outputs are consumed, preserving the ordering edge.
                        previous, source_cache, current = lax.optimization_barrier(
                            (previous, cache, current)
                        )
                        output = append_output(output, previous, (batch_id - 1) * bq_sparse)
                    next_cache = gather_separate(batch_id + 1, source_cache)
                    previous = attend(batch_id, current)
                    current = next_cache
            final_batch = num_microbatches - 1
            if previous is not None:
                previous, current = lax.optimization_barrier((previous, current))
                output = append_output(output, previous, (final_batch - 1) * bq_sparse)
            final = attend(final_batch, current)
            output = append_output(output, final, final_batch * bq_sparse)
            if return_lse:
                return jax.tree.map(lambda *parts: jnp.concatenate(parts, axis=0), *output)
            return output

        # Consume two microbatches per iteration without a conditional merge of
        # two large cache arrays. The previous parity branches caused XLA to
        # insert same-layout cache copies in the full model.
        def pair_step(pair_id, carry):
            current, output = carry
            with jax.named_scope("pipeline_first"):
                current, output = consume(pair_id * 2, current, output)
            with jax.named_scope("pipeline_second"):
                return consume(pair_id * 2 + 1, current, output)

        pair_count = (num_microbatches - 1) // 2
        current, output = lax.fori_loop(0, pair_count, pair_step, (initial, empty_output()))
        if (num_microbatches - 1) % 2:
            current, output = consume(pair_count * 2, current, output)
        final_batch = num_microbatches - 1
        final = attend(final_batch, current)
        return update_output(output, final, final_batch * bq_sparse)

    def gather_into(batch_id, output_pool, output_buffer_index):
        return _gather_cache_microbatch_into(
            cache,
            lax.dynamic_index_in_dim(safe_slot_batches, batch_id, keepdims=False),
            output_pool,
            output_buffer_index,
        )

    cache_buffers = jnp.empty(
        (2, bq_sparse, topk, cache.shape[-2], cache.shape[-1]),
        dtype=cache.dtype,
    )
    cache_buffers = gather_into(0, cache_buffers, jnp.int32(0))
    output = empty_output()

    def pipeline_step(batch_id, carry):
        cache_buffers, output_buffer = carry
        current_buffer_index = lax.bitwise_and(batch_id, jnp.int32(1))
        next_buffer_index = lax.bitwise_xor(current_buffer_index, jnp.int32(1))
        cache_buffers = gather_into(
            batch_id + 1,
            cache_buffers,
            next_buffer_index,
        )
        current_output = _tensor_core_attention(
            lax.dynamic_index_in_dim(q_latent_batches, batch_id, keepdims=False),
            lax.dynamic_index_in_dim(q_rope_batches, batch_id, keepdims=False),
            cache_buffers,
            lax.dynamic_index_in_dim(count_batches, batch_id, keepdims=False),
            current_buffer_index,
            sm_scale=sm_scale,
            bq=bq,
            b_topk=b_topk,
            fp8_dtype=fp8_dtype,
            fp8_q=fp8_q,
            fp8_pv=fp8_pv,
            return_lse=return_lse,
        )
        output_buffer = update_output(
            output_buffer,
            current_output,
            batch_id * bq_sparse,
        )
        return cache_buffers, output_buffer

    cache_buffers, output = lax.fori_loop(
        0,
        num_microbatches - 1,
        pipeline_step,
        (cache_buffers, output),
    )
    final_batch = num_microbatches - 1
    final_output = _tensor_core_attention(
        lax.dynamic_index_in_dim(q_latent_batches, final_batch, keepdims=False),
        lax.dynamic_index_in_dim(q_rope_batches, final_batch, keepdims=False),
        cache_buffers,
        lax.dynamic_index_in_dim(count_batches, final_batch, keepdims=False),
        jnp.int32(final_batch & 1),
        sm_scale=sm_scale,
        bq=bq,
        b_topk=b_topk,
        fp8_dtype=fp8_dtype,
        fp8_q=fp8_q,
        fp8_pv=fp8_pv,
        return_lse=return_lse,
    )
    return update_output(
        output,
        final_output,
        final_batch * bq_sparse,
    )
