# SPDX-License-Identifier: Apache-2.0
# Verbatim copy of primatrix-glm/sglang-jax-private (Apache-2.0) at c948bdda95208e68a474c8088937347d08fe8937:
#   python/sgl_jax/srt/kernels/dsa/gather.py
# Only change: import prefix 'sgl_jax.srt.' -> 'sglang_tpu.attention.pallas.source_dsa.'. See source_dsa/MANIFEST.json.
"""Shared SparseCore gather for exact DSA pipelines.

Keep Pallas call names, layouts, aliases and cost metadata stable: the HLO
scheduling pass and downstream profiling consume these contracts.
"""

import functools
import inspect

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.fp8_kv_cache import (
    FP8_KV_CACHE_NUM_LANES,
    GatheredFP8KVGroups,
    SplitFP8KVRows,
)

_SC_GATHER_WINDOW = 128


def _align_to(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _sparse_core_info():
    """Return SparseCore target information while tracing on a TPU."""
    info = pltpu.get_tpu_info().sparse_core
    if info is None:
        raise RuntimeError("The current TPU target does not expose SparseCore support.")
    return info


def _make_sparse_core_mesh(info):
    return plsc.VectorSubcoreMesh(
        num_cores=info.num_cores,
        num_subcores=info.num_subcores,
        core_axis_name="sc_core",
        subcore_axis_name="sc_subcore",
    )


def _sparse_core_gather_cost_estimate(
    num_indices: int,
    pair_dim: int,
    value_dim: int,
    *,
    table_dtype: jnp.dtype,
    indices_dtype: jnp.dtype,
    output_dtype: jnp.dtype,
) -> pl.CostEstimate:
    """Estimate the HBM traffic for one SparseCore gather launch.

    The table is indexed indirectly, so only the selected rows are read.  A
    gather has no arithmetic work; the value bytes are counted once for the
    HBM read and once for the HBM write.
    """
    value_elements = num_indices * pair_dim * value_dim
    bytes_accessed = (
        num_indices * jnp.dtype(indices_dtype).itemsize
        + value_elements * jnp.dtype(table_dtype).itemsize
        + value_elements * jnp.dtype(output_dtype).itemsize
    )
    return pl.CostEstimate(
        flops=0,
        bytes_accessed=bytes_accessed,
        transcendentals=0,
    )


def _sparse_core_gather(
    table: jax.Array,
    indices: jax.Array,
    output_buffer: jax.Array | None = None,
    output_buffer_index: jax.Array | int | None = None,
    *,
    skip_invalid: bool = False,
    native_byte_gather: bool = False,
) -> jax.Array:
    """Gather complete cache rows by slot index.

    Legacy TC-tiling rows use BF16 ``[2, D]`` or byte ``[4, D]`` layouts.
    With ``native_byte_gather=True``, byte rows use native SC tiling and DMA
    without a widening Ref.bitcast. Split FP8 uses ``[4, 128]`` for latent
    and grouped ``[4, 128]`` for RoPE, each with an independent gather output.
    With ``skip_invalid=True``, index -1 skips HBM reads and yields a zero row.

    If ``output_buffer`` is supplied, it is a two-slot pool and the SparseCore
    kernel writes into the selected slot through a stateful Ref. This keeps
    the pool in one fixed loop-state position.
    """
    if table.dtype not in (jnp.bfloat16, jnp.uint8) or table.ndim != 3:
        raise TypeError("SparseCore table must be rank-3 BF16/uint8.")
    if native_byte_gather and table.dtype != jnp.uint8:
        raise TypeError("Native byte gather requires uint8 FP8 storage.")
    expected_pair_dim = (
        table.shape[1]
        if native_byte_gather
        else FP8_KV_CACHE_NUM_LANES if table.dtype == jnp.uint8 else 2
    )
    if table.shape[1] != expected_pair_dim:
        raise TypeError(
            f"SparseCore {table.dtype} table must have shape "
            f"[S, {expected_pair_dim}, D], got {table.shape}."
        )
    if table.shape[-1] % 128:
        raise ValueError("SparseCore cache pair dimension D must be divisible by 128.")
    if indices.dtype != jnp.int32 or indices.ndim != 1:
        raise TypeError("SparseCore indices must be a rank-1 int32 array.")

    info = _sparse_core_info()
    num_indices = indices.shape[0]
    indices_per_step = _SC_GATHER_WINDOW * info.num_cores * info.num_subcores
    if num_indices % indices_per_step:
        raise ValueError(
            f"the number of gather indices must be divisible by "
            f"{indices_per_step}, got {num_indices}"
        )

    pair_dim = table.shape[1]
    value_dim = table.shape[-1]
    # Native SC tiling supports byte DMA directly. Legacy TC tiling uses
    # a widening view for BF16 and the combined four-lane byte layout.
    kernel_pair_dim = pair_dim if native_byte_gather else 1
    dma_dtype = table.dtype if native_byte_gather else jnp.int32
    kernel_value_dim = value_dim
    mesh = _make_sparse_core_mesh(info)

    def kernel(table_hbm, indices_hbm, output_hbm):
        table_dma = table_hbm if native_byte_gather else table_hbm.bitcast(jnp.int32)
        output_dma = output_hbm if native_byte_gather else output_hbm.bitcast(jnp.int32)
        workers = lax.axis_size((mesh.core_axis_name, mesh.subcore_axis_name))
        row_wave_size = _SC_GATHER_WINDOW * workers
        num_row_chunks = num_indices // row_wave_size
        subcore_first_row_chunk = (
            lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name)) * num_row_chunks
        )
        row_subchunk_size = info.num_lanes
        num_row_subchunks = _SC_GATHER_WINDOW // row_subchunk_size
        if num_row_chunks == 0:
            return

        num_data_banks = 8
        read_ahead = 4
        filtered_bytes = (
            num_data_banks
            * row_subchunk_size
            * kernel_pair_dim
            * kernel_value_dim
            * jnp.dtype(dma_dtype).itemsize
            + 2 * _SC_GATHER_WINDOW * 4
        )

        if skip_invalid and filtered_bytes <= info.vmem_capacity_bytes:
            # Gather and write back from the same eight VMEM banks, with four
            # reads prefetched and older writes draining independently. This
            # replaces the two read + two output banks and their VMEM copy.
            # Index banks remain live until their last indirect read completes.
            @functools.partial(
                pl.run_scoped,
                index_buffers=pltpu.VMEM((2, _SC_GATHER_WINDOW), jnp.int32),
                index_sems=pltpu.SemaphoreType.DMA((2,)),
                data_buffers=pltpu.VMEM(
                    (
                        num_data_banks,
                        row_subchunk_size,
                        kernel_pair_dim,
                        kernel_value_dim,
                    ),
                    dma_dtype,
                ),
                read_sems=pltpu.SemaphoreType.DMA((num_data_banks,)),
                write_sems=pltpu.SemaphoreType.DMA((num_data_banks,)),
            )
            def filtered_reads(index_buffers, index_sems, data_buffers, read_sems, write_sems):
                steps_per_chunk = num_row_subchunks
                num_steps = num_row_chunks * steps_per_chunk

                def index_descriptor(chunk):
                    return pltpu.make_async_copy(
                        indices_hbm.at[
                            pl.ds(
                                (subcore_first_row_chunk + chunk) * _SC_GATHER_WINDOW,
                                _SC_GATHER_WINDOW,
                            )
                        ],
                        index_buffers.at[chunk % 2],
                        index_sems.at[chunk % 2],
                    )

                def index_window(step):
                    chunk = step // steps_per_chunk
                    row_subchunk = step % num_row_subchunks
                    return index_buffers.at[
                        chunk % 2,
                        pl.ds(row_subchunk * row_subchunk_size, row_subchunk_size),
                    ]

                def read_descriptor(step, bank):
                    offsets = plsc.Indices(index_window(step), ignored_value=-1)
                    return pltpu.make_async_copy(
                        table_dma.at[offsets, :, :],
                        data_buffers.at[bank],
                        read_sems.at[bank],
                    )

                def write_descriptor(step, bank):
                    first_row = (
                        subcore_first_row_chunk * _SC_GATHER_WINDOW + step * row_subchunk_size
                    )
                    return pltpu.make_async_copy(
                        data_buffers.at[bank],
                        output_dma.at[pl.ds(first_row, row_subchunk_size), :, :],
                        write_sems.at[bank],
                    )

                def start_read(step, bank):
                    # Legal offsets are nonnegative or -1, so their bitwise
                    # OR is -1 exactly when an ignored slot is present. Avoid
                    # reductions that SparseCore paired-layout inference rejects.
                    offsets = index_window(step)[...]
                    combined = offsets[0]
                    for lane in range(1, row_subchunk_size):
                        combined = combined | offsets[lane]

                    @pl.when(combined == -1)
                    def clear_ignored_rows():
                        data_buffers[bank] = jnp.zeros(
                            (row_subchunk_size, kernel_pair_dim, kernel_value_dim),
                            dma_dtype,
                        )

                    read_descriptor(step, bank).start()

                for chunk in range(min(2, num_row_chunks)):
                    index_descriptor(chunk).start()
                index_descriptor(0).wait()
                for step in range(min(read_ahead, num_steps)):
                    start_read(step, step)

                def transfer_step(step, _):
                    chunk = step // steps_per_chunk
                    chunk_step = step % steps_per_chunk
                    bank = step % num_data_banks
                    read_descriptor(step, bank).wait()
                    write_descriptor(step, bank).start()

                    @pl.when((chunk_step == steps_per_chunk - 1) & (chunk + 2 < num_row_chunks))
                    def refill_indices():
                        index_descriptor(chunk + 2).start()

                    @pl.when(step + read_ahead < num_steps)
                    def prefetch():
                        next_step = step + read_ahead
                        next_bank = next_step % num_data_banks

                        # A bank's previous HBM write must finish before either
                        # zeroing ignored rows or starting the next gather.
                        @pl.when(next_step >= num_data_banks)
                        def wait_old_write():
                            write_descriptor(next_step - num_data_banks, next_bank).wait()

                        @pl.when(next_step % steps_per_chunk == 0)
                        def wait_indices():
                            index_descriptor(next_step // steps_per_chunk).wait()

                        start_read(next_step, next_bank)

                    return None

                lax.fori_loop(0, num_steps, transfer_step, None, unroll=False)

                # Earlier writes were waited on when their bank was reused.
                for step in range(max(0, num_steps - num_data_banks), num_steps):
                    write_descriptor(step, step % num_data_banks).wait()

            return

        @functools.partial(
            pltpu.emit_pipeline,
            grid=(num_row_chunks,),
            in_specs=pl.BlockSpec(
                (_SC_GATHER_WINDOW,),
                lambda chunk: (subcore_first_row_chunk + chunk,),
            ),
        )
        def index_pipeline(indices_vmem):
            row_chunk = subcore_first_row_chunk + pl.program_id(0)

            if skip_invalid:

                @functools.partial(
                    pltpu.emit_pipeline,
                    grid=(num_row_subchunks,),
                    out_specs=pl.BlockSpec(
                        (row_subchunk_size, kernel_pair_dim, kernel_value_dim),
                        lambda row_subchunk: (
                            row_chunk * num_row_subchunks + row_subchunk,
                            0,
                            0,
                        ),
                    ),
                )
                def filtered_pipeline(output_vmem):
                    # Filtered DMA leaves ignored destinations untouched.
                    # Clear them so stale data/NaNs cannot reach attention.
                    output_vmem[...] = jnp.zeros(output_vmem.shape, dma_dtype)
                    offsets = plsc.Indices(
                        indices_vmem.at[
                            pl.ds(pl.program_id(0) * row_subchunk_size, row_subchunk_size)
                        ],
                        ignored_value=-1,
                    )
                    pltpu.sync_copy(table_dma.at[offsets, :, :], output_vmem)

                filtered_pipeline(output_dma)
                return

            @functools.partial(
                pltpu.emit_pipeline,
                grid=(num_row_subchunks, 1),
                in_specs=pl.BlockSpec(
                    (
                        pl.Indirect(row_subchunk_size),
                        kernel_pair_dim,
                        kernel_value_dim,
                    ),
                    lambda row_subchunk, _col: (
                        indices_vmem[
                            pl.ds(
                                row_subchunk * row_subchunk_size,
                                row_subchunk_size,
                            )
                        ],
                        0,
                        0,
                    ),
                ),
                out_specs=pl.BlockSpec(
                    (row_subchunk_size, kernel_pair_dim, kernel_value_dim),
                    lambda row_subchunk, _col: (
                        row_chunk * num_row_subchunks + row_subchunk,
                        0,
                        0,
                    ),
                ),
            )
            def data_pipeline(gather_vmem, output_vmem):
                @plsc.parallel_loop(0, kernel_value_dim, step=32)
                def copy_columns(column):
                    output_vmem[
                        pl.ds(0, row_subchunk_size),
                        pl.ds(0, kernel_pair_dim),
                        pl.ds(column, 32),
                    ] = gather_vmem[
                        pl.ds(0, row_subchunk_size),
                        pl.ds(0, kernel_pair_dim),
                        pl.ds(column, 32),
                    ]

            data_pipeline(
                table_dma,
                output_dma,
            )

        index_pipeline(indices_hbm)

    output = jax.ShapeDtypeStruct(
        (num_indices, pair_dim, value_dim),
        table.dtype,
    )
    compiler_param_kwargs = {"use_tc_tiling_on_sc": not native_byte_gather}
    if "needs_layout_passes" in inspect.signature(pltpu.CompilerParams).parameters:
        compiler_param_kwargs["needs_layout_passes"] = True
    kwargs = {
        "mesh": mesh,
        "scratch_types": [],
        "compiler_params": pltpu.CompilerParams(**compiler_param_kwargs),
        "name": "dsa_sparse_core_gather",
    }
    # JAX 0.10 called this argument out_shape; JAX 0.11 calls it out_type.
    if "out_type" in inspect.signature(pl.kernel).parameters:
        kwargs["out_type"] = output
    else:
        kwargs["out_shape"] = output
        kwargs["scratch_shapes"] = kwargs.pop("scratch_types")
    kwargs["cost_estimate"] = _sparse_core_gather_cost_estimate(
        num_indices,
        pair_dim,
        value_dim,
        table_dtype=table.dtype,
        indices_dtype=indices.dtype,
        output_dtype=table.dtype,
    )
    if output_buffer is None:
        if output_buffer_index is not None:
            raise ValueError("output_buffer_index requires output_buffer.")
        return pl.kernel(kernel, **kwargs)(table, indices)

    if native_byte_gather:
        raise ValueError("Native byte gather uses independent outputs, not an aliased pool.")
    if (
        output_buffer.dtype != table.dtype
        or output_buffer.ndim < 2
        or output_buffer.shape[0] != 2
        or output_buffer.size != 2 * num_indices * pair_dim * value_dim
    ):
        raise TypeError(
            "SparseCore gather output pool must have leading dimension 2 and "
            f"two slots of size {output.shape}, got "
            f"{output_buffer.shape} and {output_buffer.dtype}."
        )
    if output_buffer_index is None:
        raise ValueError("output_buffer_index is required for an output pool.")

    compiler_params = kwargs["compiler_params"]
    cost_estimate = kwargs["cost_estimate"]

    def run_stateful_gather(output_slot):
        def stateful_gather(refs):
            table_ref, indices_ref, output_ref = refs

            def run_kernel():
                kernel(
                    table_ref,
                    indices_ref,
                    output_ref.at[output_slot].reshape(-1, pair_dim, value_dim),
                )

            pl.core_map(
                mesh,
                compiler_params=compiler_params,
                cost_estimate=cost_estimate,
                name="dsa_sparse_core_gather_into",
            )(run_kernel)

        _, _, updated_buffer = pl.run_state(stateful_gather)((table, indices, output_buffer))
        return updated_buffer

    return lax.cond(
        output_buffer_index == 0,
        lambda _: run_stateful_gather(0),
        lambda _: run_stateful_gather(1),
        operand=None,
    )


def _gather_split_fp8(cache, slots, *, fused=False):
    """Gather compact token rows from latent and grouped RoPE storage."""
    if fused:
        return _sparse_core_gather_split_bytes(cache.rope, slots, latent=cache.latent)
    return SplitFP8KVRows(
        _sparse_core_gather(cache.latent, slots, skip_invalid=True),
        _sparse_core_gather_rope(cache.rope, slots),
    )


def _gather_split_fp8_groups(cache, slots):
    """Gather all four adjacent RoPE rows; TensorCore selects the target byte."""
    return GatheredFP8KVGroups(
        _sparse_core_gather(cache.latent, slots, skip_invalid=True),
        _sparse_core_gather(cache.rope, slots >> 2, skip_invalid=True),
        slots & 3,
    )


def _sparse_core_gather_rope(table, indices):
    return _sparse_core_gather_split_bytes(table, indices)


def _sparse_core_gather_split_bytes(table, indices, *, latent=None):
    """Gather compact token rows directly from [S//4, 4, D] byte storage.

    TC packing interleaves four adjacent byte rows into each uint32 row.
    A widening Ref view reads the group containing each selected row. SC
    extracts its byte and interleaves four output rows in VMEM. Persistent
    storage uses four-token groups; the returned uint8 rows have one D feature axis.
    """
    if table.dtype != jnp.uint8 or table.ndim != 3 or table.shape[1] != 4 or table.shape[-1] % 128:
        raise ValueError("RoPE storage must use uint8 [S//4, 4, D] with D divisible by 128.")
    if indices.dtype != jnp.int32 or indices.ndim != 1:
        raise TypeError("SparseCore indices must be a rank-1 int32 array.")
    fused = latent is not None
    if fused and (
        latent.dtype != jnp.uint8
        or latent.ndim != 3
        or latent.shape[:2] != (table.shape[0] * 4, 4)
        or latent.shape[-1] % 128
    ):
        raise ValueError("Latent storage must use uint8 [S, 4, D] with D divisible by 128.")
    info = _sparse_core_info()
    dim = table.shape[-1]
    count = indices.shape[0]
    if count % (_SC_GATHER_WINDOW * info.num_cores * info.num_subcores):
        raise ValueError("RoPE gather index count must be divisible by the SparseCore window.")
    mesh = _make_sparse_core_mesh(info)
    lanes = info.num_lanes
    if lanes % 4:
        raise ValueError("RoPE gather requires a SparseCore lane count divisible by four.")
    workers = info.num_cores * info.num_subcores
    chunks = count // (_SC_GATHER_WINDOW * workers)
    subchunks = _SC_GATHER_WINDOW // lanes
    steps = chunks * subchunks
    # A fused step issues both latent and RoPE reads. Keep three steps in
    # flight, with separate write banks, to bound DMA and scratch pressure.
    banks, ahead = (6, 3) if fused else (8, 4)

    def kernel(*refs):
        if fused:
            latent_src, src, ids, latent_dst, dst = refs
            latent_src, latent_dst = latent_src.bitcast(jnp.uint32), latent_dst.bitcast(jnp.uint32)
        else:
            src, ids, dst = refs
        # The persistent group axis packs four adjacent tokens into each word.
        src, dst = src.bitcast(jnp.uint32), dst.bitcast(jnp.uint32)
        first_chunk = lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name)) * chunks
        latent_scratch = (
            dict(
                latent_data=pltpu.VMEM((banks, lanes, 1, latent.shape[-1]), jnp.uint32),
                latent_read_sems=pltpu.SemaphoreType.DMA((banks,)),
                latent_write_sems=pltpu.SemaphoreType.DMA((banks,)),
            )
            if fused
            else {}
        )

        @functools.partial(
            pl.run_scoped,
            index=pltpu.VMEM((2, _SC_GATHER_WINDOW), jnp.int32),
            index_sems=pltpu.SemaphoreType.DMA((2,)),
            groups=pltpu.VMEM((banks, lanes), jnp.int32),
            shifts=pltpu.VMEM((banks, lanes), jnp.int32),
            data=pltpu.VMEM((banks, lanes, 1, dim), jnp.uint32),
            packed=pltpu.VMEM((banks, lanes // 4, dim), jnp.uint32),
            read_sems=pltpu.SemaphoreType.DMA((banks,)),
            write_sems=pltpu.SemaphoreType.DMA((banks,)),
            **latent_scratch,
        )
        def gather(
            index,
            index_sems,
            groups,
            shifts,
            data,
            packed,
            read_sems,
            write_sems,
            latent_data=None,
            latent_read_sems=None,
            latent_write_sems=None,
        ):
            def fetch_indices(chunk):
                return pltpu.make_async_copy(
                    ids.at[pl.ds((first_chunk + chunk) * _SC_GATHER_WINDOW, _SC_GATHER_WINDOW)],
                    index.at[chunk % 2],
                    index_sems.at[chunk % 2],
                )

            def read_latent(step, bank):
                offsets = index.at[step // subchunks % 2, pl.ds(step % subchunks * lanes, lanes)]
                return pltpu.make_async_copy(
                    latent_src.at[plsc.Indices(offsets, ignored_value=-1), :, :],
                    latent_data.at[bank],
                    latent_read_sems.at[bank],
                )

            def write_latent(step, bank):
                first_row = first_chunk * _SC_GATHER_WINDOW + step * lanes
                return pltpu.make_async_copy(
                    latent_data.at[bank],
                    latent_dst.at[pl.ds(first_row, lanes), :, :],
                    latent_write_sems.at[bank],
                )

            def read(bank):
                return pltpu.make_async_copy(
                    src.at[plsc.Indices(groups.at[bank], ignored_value=-1), :, :],
                    data.at[bank],
                    read_sems.at[bank],
                )

            def write(step, bank):
                first_row = first_chunk * (_SC_GATHER_WINDOW // 4) + step * (lanes // 4)
                return pltpu.make_async_copy(
                    packed.at[bank],
                    dst.at[pl.ds(first_row, lanes // 4), :],
                    write_sems.at[bank],
                )

            def start_read(step, bank):
                offsets = index[step // subchunks % 2, pl.ds(step % subchunks * lanes, lanes)]
                # Arithmetic shift maps the ignored -1 sentinel to -1, too.
                # A vector select/divide here triggers SC paired-layout bugs.
                groups[bank] = offsets >> 2
                shifts[bank] = (offsets & 3) * 8
                combined = offsets[0]
                for lane in range(1, lanes):
                    combined = combined | offsets[lane]

                @pl.when(combined == -1)
                def clear_ignored():
                    data[bank] = jnp.zeros((lanes, 1, dim), jnp.uint32)
                    if fused:
                        latent_data[bank] = jnp.zeros((lanes, 1, latent.shape[-1]), jnp.uint32)

                if fused:
                    read_latent(step, bank).start()
                read(bank).start()

            for chunk in range(min(2, chunks)):
                fetch_indices(chunk).start()
            fetch_indices(0).wait()
            for step in range(min(ahead, steps)):
                start_read(step, step)

            def transfer(step, _):
                bank = step % banks
                if fused:
                    read_latent(step, bank).wait()
                read(bank).wait()
                row_shifts = shifts[bank]

                @plsc.parallel_loop(0, dim, step=32)
                def pack_columns(column):
                    for row in range(lanes // 4):
                        words = jnp.zeros((32,), jnp.uint32)
                        for lane in range(4):
                            shift = row_shifts[row * 4 + lane].astype(jnp.uint32)
                            byte = (
                                data[bank, row * 4 + lane, 0, pl.ds(column, 32)] >> shift
                            ) & jnp.uint32(255)
                            words = words | (byte << (lane * 8))
                        packed[bank, row, pl.ds(column, 32)] = words

                if fused:
                    write_latent(step, bank).start()
                write(step, bank).start()
                chunk = step // subchunks

                @pl.when((step % subchunks == subchunks - 1) & (chunk + 2 < chunks))
                def refill_indices():
                    fetch_indices(chunk + 2).start()

                @pl.when(step + ahead < steps)
                def prefetch():
                    next_step = step + ahead
                    next_bank = next_step % banks

                    @pl.when(next_step >= banks)
                    def wait_write():
                        if fused:
                            write_latent(next_step - banks, next_bank).wait()
                        write(next_step - banks, next_bank).wait()

                    @pl.when(next_step % subchunks == 0)
                    def wait_indices():
                        fetch_indices(next_step // subchunks).wait()

                    start_read(next_step, next_bank)

                return None

            lax.fori_loop(0, steps, transfer, None, unroll=False)
            for step in range(max(0, steps - banks), steps):
                if fused:
                    write_latent(step, step % banks).wait()
                write(step, step % banks).wait()

    compiler_kwargs = {"use_tc_tiling_on_sc": True}
    if "needs_layout_passes" in inspect.signature(pltpu.CompilerParams).parameters:
        compiler_kwargs["needs_layout_passes"] = True
    kwargs = dict(
        mesh=mesh,
        scratch_types=[],
        compiler_params=pltpu.CompilerParams(**compiler_kwargs),
        name="dsa_sparse_core_gather",
        cost_estimate=pl.CostEstimate(
            flops=0,
            bytes_accessed=count * (5 * dim + (8 * latent.shape[-1] if fused else 0) + 4),
            transcendentals=0,
        ),
    )
    output = jax.ShapeDtypeStruct((count, dim), jnp.uint8)
    if fused:
        output = (jax.ShapeDtypeStruct((count, 4, latent.shape[-1]), jnp.uint8), output)
    if "out_type" in inspect.signature(pl.kernel).parameters:
        kwargs["out_type"] = output
    else:
        kwargs["out_shape"] = output
        kwargs["scratch_shapes"] = kwargs.pop("scratch_types")
    if fused:
        return SplitFP8KVRows(*pl.kernel(kernel, **kwargs)(latent, table, indices))
    return pl.kernel(kernel, **kwargs)(table, indices)
