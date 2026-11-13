# SPDX-License-Identifier: Apache-2.0
# Adapted from primatrix-glm/sglang-jax-private (Apache-2.0)
# python/sgl_jax/srt/kernels/dsa/fused_fp8_k_projection_scatter.py at c948bdd.
"""DSA indexer K prologue for page-aligned prefill, as one Pallas kernel.

For each 128-token page of this step, one kernel launch computes what
``dsa_indexer_tpu.indexer_forward`` and ``dsa_indexer_step(key_store="xla")``
compute for the index keys, and writes the page's records:

1. ``wk``: per-row FP8 activation quantization, FP8 dot with f32 accumulation,
   activation and channel scales after the dot, bf16 output
   (``fp8_linear.apply_tpu_fp8_linear``).
2. ``k_norm``: LayerNorm in f32, bf16 output.
3. Interleaved (GPT-J) rope on the first ``rope_dim`` dims, bf16 output.
4. UE8M0 FP8 quantization and the index-key record (``index_key_cache_shape``:
   128 key bytes, one scale byte, zeros).

Differences from the source kernel: the cache is the target's FP8 record cache
(page 128) instead of a BF16 cache (page 64); there is no Hadamard rotation;
the activations are quantized to FP8 as the target's linear does; LayerNorm eps
is the target module's.

Only page-aligned batches can use it: every real request's new tokens start on
a page boundary and fill whole pages, so each written page holds this step's
tokens only. ``key_prologue_targets`` reports whether a batch qualifies.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.attention.pallas.dsa_key_store import _ue8m0_quantize

_FP8_MAX = 448.0
_TOKENS_PER_ROW = 4
_HEAD_DIM = 128
_RECORD_BYTES = 256


def _bf16(a):
    """Round f32 to bf16 values, kept in f32.

    torch_tpu holds real bf16 tensors between ops; XLA folds an
    astype(bf16).astype(f32) pair away, so the rounding is explicit here.
    """
    return lax.reduce_precision(a, exponent_bits=8, mantissa_bits=7)


def key_prologue_targets(
    num_tokens: int,
    kv_lens: jax.Array,  # i32[S]; already counts this step's tokens
    page_indices: jax.Array,  # i32[S * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[S + 1]
    distribution: jax.Array,  # i32[3]; [2] is the number of real requests
    page_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Cache page of each token page of the batch (-1: not stored), and whether
    the batch is page-aligned so the kernel may write whole pages."""
    num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // num_seqs
    # Tokens after the real requests are batch padding, as in _store_keys.
    active = jnp.minimum(cu_q_lens, cu_q_lens[distribution[2]])
    num_real = active[-1]

    start = jnp.arange(num_tokens // page_size, dtype=jnp.int32) * page_size
    seq = jnp.clip(jnp.searchsorted(active[1:], start, side="right"), 0, num_seqs - 1)
    q_len = active[seq + 1] - active[seq]
    pos = kv_lens[seq] - q_len + (start - active[seq])
    page_idx = jnp.clip(pos // page_size, 0, pages_per_seq - 1)
    page = page_indices[seq * pages_per_seq + page_idx]
    targets = jnp.where(start < num_real, page, -1).astype(jnp.int32)

    real_seq = jnp.arange(num_seqs) < distribution[2]
    lens = active[1:] - active[:-1]
    prefix = kv_lens - lens
    seq_ok = (lens % page_size == 0) & (prefix % page_size == 0) & (prefix >= 0)
    aligned = (
        (num_tokens % page_size == 0)
        & (num_real % page_size == 0)
        & jnp.all(jnp.where(real_seq, seq_ok, True))
    )
    return targets, aligned


def _key_stages(x, w, ws, nw, nb, cos, sin, *, eps):
    """One token page's (x_scale, projected, normed, roped) keys, in f32 holding bf16 values.

    Traced inside the kernel; also used by the stage diagnostic kernel.
    """
    x = x.astype(jnp.float32)
    x_scale = jnp.maximum(jnp.max(jnp.abs(x), axis=1, keepdims=True) / _FP8_MAX, 1e-12)
    x_q = (x / x_scale).astype(jnp.float8_e4m3fn)
    # Not bit-exact with the target's XLA FP8 dot: on v7x both lose ~0.5 (mean)
    # on |acc| ~ 800, in different elements. test_dsa_key_prologue.py bounds it.
    acc = lax.dot_general(x_q, w, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    k_proj = _bf16(acc * x_scale * ws)

    mean = jnp.mean(k_proj, axis=1, keepdims=True)
    var = jnp.mean(jnp.square(k_proj - mean), axis=1, keepdims=True)
    k_norm = _bf16((k_proj - mean) * lax.rsqrt(var + eps) * nw + nb)

    # GPT-J rope: lane 2j pairs with 2j + 1. roll(k, s)[l] == k[l - s].
    lane = lax.broadcasted_iota(jnp.int32, k_norm.shape, 1)
    partner = jnp.where(
        lane % 2 == 0, -pltpu.roll(k_norm, _HEAD_DIM - 1, 1), pltpu.roll(k_norm, 1, 1)
    )
    # Round after each op as the eager bf16 rope does (rotary_embedding/utils.py).
    cos = cos.astype(jnp.float32)
    sin = sin.astype(jnp.float32)
    k_rope = _bf16(_bf16(k_norm * cos) + _bf16(partner * sin))
    return x_scale, k_proj, k_norm, k_rope


def key_chain_xla(hidden, wk, wk_scale, norm_weight, norm_bias, rope_cos, rope_sin, *, eps):
    """The kernel's key chain in plain XLA, for batches the kernel cannot take.

    Same math as ``_key_stages`` (XLA's native FP8 dot), on any number of
    tokens; returns bf16[T, 128], the key ``quantize_fp8_ue8m0`` takes.
    """
    f32 = jnp.float32
    x = hidden.astype(f32)
    x_scale = jnp.maximum(jnp.max(jnp.abs(x), axis=1, keepdims=True) / _FP8_MAX, 1e-12)
    x_q = (x / x_scale).astype(jnp.float8_e4m3fn)
    acc = lax.dot_general(x_q, wk, (((1,), (0,)), ((), ())), preferred_element_type=f32)
    k = _bf16(acc * x_scale * wk_scale.astype(f32)[None])
    mean = jnp.mean(k, axis=1, keepdims=True)
    var = jnp.mean(jnp.square(k - mean), axis=1, keepdims=True)
    k = _bf16((k - mean) * lax.rsqrt(var + eps) * norm_weight.astype(f32) + norm_bias.astype(f32))
    rope_dim = 2 * rope_cos.shape[1]
    c, s = rope_cos.astype(f32), rope_sin.astype(f32)
    x1, x2 = k[:, 0:rope_dim:2], k[:, 1:rope_dim:2]
    o1 = _bf16(_bf16(x1 * c) - _bf16(x2 * s))
    o2 = _bf16(_bf16(x2 * c) + _bf16(x1 * s))
    rot = jnp.stack([o1, o2], axis=-1).reshape(k.shape[0], rope_dim)
    return jnp.concatenate([rot, k[:, rope_dim:]], axis=1).astype(jnp.bfloat16)


def key_stages_pallas(
    hidden, wk, wk_scale, norm_weight, norm_bias, rope_cos, rope_sin, *, eps, interpret=False
):
    """Diagnostic: the kernel's per-stage values, f32[T, 128] each (x_scale f32[T, 128], lane 0).

    Same math as key_prologue_pallas, compiled by Mosaic; for comparing stages with torch.
    """
    num_tokens, hidden_dim = hidden.shape
    page_size = 128
    half = rope_cos.shape[1]
    pad = ((0, 0), (0, _HEAD_DIM - 2 * half))
    cos = jnp.pad(jnp.repeat(rope_cos.astype(jnp.bfloat16), 2, axis=1), pad, constant_values=1)
    sin = jnp.pad(jnp.repeat(rope_sin.astype(jnp.bfloat16), 2, axis=1), pad)
    row = lambda a: a.astype(jnp.float32).reshape(1, _HEAD_DIM)

    def kernel(x_ref, w_ref, ws_ref, nw_ref, nb_ref, c_ref, s_ref, xs_o, p_o, n_o, r_o):
        xs, p, n, r = _key_stages(
            x_ref[...], w_ref[...], ws_ref[...], nw_ref[...], nb_ref[...],
            c_ref[...], s_ref[...], eps=eps,
        )
        xs_o[...] = jnp.broadcast_to(xs, xs_o.shape)
        p_o[...], n_o[...], r_o[...] = p, n, r

    whole = lambda shape: pl.BlockSpec(shape, lambda i: (0, 0))
    tokens = lambda width: pl.BlockSpec((page_size, width), lambda i: (i, 0))
    out = jax.ShapeDtypeStruct((num_tokens, _HEAD_DIM), jnp.float32)
    return pl.pallas_call(
        kernel,
        out_shape=(out, out, out, out),
        grid=(num_tokens // page_size,),
        in_specs=[
            tokens(hidden_dim), whole((hidden_dim, _HEAD_DIM)), whole((1, _HEAD_DIM)),
            whole((1, _HEAD_DIM)), whole((1, _HEAD_DIM)), tokens(_HEAD_DIM), tokens(_HEAD_DIM),
        ],
        out_specs=[tokens(_HEAD_DIM)] * 4,
        interpret=interpret,
        name="dsa_index_key_prologue_stages",
    )(hidden, wk, row(wk_scale), row(norm_weight), row(norm_bias), cos, sin)


def key_proj_probe_pallas(hidden, wk, inv_scale, xq_ext, *, interpret=False):
    """Diagnostic: the FP8 projection inside Mosaic, split to find where it differs.

    ``inv_scale`` f32[T, 128] (1 / x_scale, broadcast) and ``xq_ext`` f8[T, hidden]
    are computed outside the kernel. Returns f32 arrays:
    (xq_div, acc_div, xq_mul, acc_mul, acc_ext): x_q by in-kernel division and its
    accumulator; x_q by multiplying the given inverse scale and its accumulator;
    the accumulator of the given x_q (tests the dot alone).
    """
    num_tokens, hidden_dim = hidden.shape
    page_size = 128
    dot = lambda q, w: lax.dot_general(
        q, w, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )

    def kernel(x_ref, w_ref, inv_ref, xe_ref, xqd_o, accd_o, xqm_o, accm_o, acce_o):
        x = x_ref[...].astype(jnp.float32)
        w = w_ref[...]
        xs = jnp.maximum(jnp.max(jnp.abs(x), axis=1, keepdims=True) / _FP8_MAX, 1e-12)
        xq_div = (x / xs).astype(jnp.float8_e4m3fn)
        xq_mul = (x * inv_ref[...][:, :1]).astype(jnp.float8_e4m3fn)
        xqd_o[...] = xq_div.astype(jnp.float32)
        accd_o[...] = dot(xq_div, w)
        xqm_o[...] = xq_mul.astype(jnp.float32)
        accm_o[...] = dot(xq_mul, w)
        acce_o[...] = dot(xe_ref[...], w)

    whole = lambda shape: pl.BlockSpec(shape, lambda i: (0, 0))
    tokens = lambda width: pl.BlockSpec((page_size, width), lambda i: (i, 0))
    wide = jax.ShapeDtypeStruct((num_tokens, hidden_dim), jnp.float32)
    narrow = jax.ShapeDtypeStruct((num_tokens, _HEAD_DIM), jnp.float32)
    return pl.pallas_call(
        kernel,
        out_shape=(wide, narrow, wide, narrow, narrow),
        grid=(num_tokens // page_size,),
        in_specs=[
            tokens(hidden_dim), whole((hidden_dim, _HEAD_DIM)),
            tokens(_HEAD_DIM), tokens(hidden_dim),
        ],
        out_specs=[
            tokens(hidden_dim), tokens(_HEAD_DIM), tokens(hidden_dim),
            tokens(_HEAD_DIM), tokens(_HEAD_DIM),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 * 1024 * 1024),
        interpret=interpret,
        name="dsa_index_key_proj_probe",
    )(hidden, wk, inv_scale, xq_ext)


_DOT_VARIANTS = ("default", "highest", "bf16_operands", "f32_operands", "k_chunked_512")


def key_dot_variant_pallas(xq, wk, variant, *, interpret=False):
    """Diagnostic: the FP8 dot xq @ wk inside Mosaic, in the form ``variant``.

    One kernel per variant, so a form Mosaic rejects does not hide the others.
    Returns the f32[T, 128] accumulator.
    """
    num_tokens, hidden_dim = xq.shape
    page_size = 128
    dims = (((1,), (0,)), ((), ()))
    f32 = jnp.float32

    def kernel(x_ref, w_ref, o_ref):
        x, w = x_ref[...], w_ref[...]
        dot = lambda a, b, **kw: lax.dot_general(a, b, dims, preferred_element_type=f32, **kw)
        if variant == "default":
            r = dot(x, w)
        elif variant == "highest":
            r = dot(x, w, precision=lax.Precision.HIGHEST)
        elif variant == "bf16_operands":
            r = dot(x.astype(jnp.bfloat16), w.astype(jnp.bfloat16))
        elif variant == "f32_operands":
            r = dot(x.astype(f32), w.astype(f32), precision=lax.Precision.HIGHEST)
        elif variant == "k_chunked_512":
            r = jnp.zeros((page_size, _HEAD_DIM), f32)
            for k0 in range(0, hidden_dim, 512):
                r = r + dot(x[:, k0 : k0 + 512], w[k0 : k0 + 512, :])
        else:
            raise ValueError(variant)
        o_ref[...] = r

    tokens = lambda width: pl.BlockSpec((page_size, width), lambda i: (i, 0))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((num_tokens, _HEAD_DIM), f32),
        grid=(num_tokens // page_size,),
        in_specs=[tokens(hidden_dim), pl.BlockSpec((hidden_dim, _HEAD_DIM), lambda i: (0, 0))],
        out_specs=tokens(_HEAD_DIM),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 * 1024 * 1024),
        interpret=interpret,
        name=f"dsa_index_key_dot_{variant}",
    )(xq, wk)


def _kernel(
    targets_ref,  # SMEM i32[num_token_pages]
    x_ref,  # VMEM bf16[page_size, hidden]
    w_ref,  # VMEM f8e4m3fn[hidden, 128]
    ws_ref,  # VMEM f32[1, 128]
    nw_ref,  # VMEM f32[1, 128]
    nb_ref,  # VMEM f32[1, 128]
    cos_ref,  # VMEM bf16[page_size, 128]; 1 past rope_dim
    sin_ref,  # VMEM bf16[page_size, 128]; 0 past rope_dim
    cache_in_ref,  # HBM, aliased with cache_ref
    cache_ref,  # HBM u8[P, page_size // 4, 4, 256]
    key_rec_ref,  # VMEM u32[page_size, 128]: record lanes [0, 128), one byte per lane
    tail_rec_ref,  # VMEM u32[page_size, 128]: record lanes [128, 256)
    page_buf,  # VMEM u8[2, page_size // 4, 4, 256]
    sems,  # DMA[2]
    pending,  # SMEM i32[2]
    *,
    eps: float,
):
    del cache_in_ref
    i = pl.program_id(0)
    slot = i % 2
    rows_per_page = page_buf.shape[1]

    def _write(s, page):
        return pltpu.make_async_copy(page_buf.at[s], cache_ref.at[page], sems.at[s])

    @pl.when(i == 0)
    def _():
        pending[0] = 0
        pending[1] = 0

    # Reuse this slot's buffer only after the write issued two pages ago landed.
    @pl.when(pending[slot] == 1)
    def _():
        _write(slot, 0).wait()
        pending[slot] = 0

    page = targets_ref[i]

    @pl.when(page >= 0)
    def _():
        _, _, _, k = _key_stages(
            x_ref[...], w_ref[...], ws_ref[...], nw_ref[...], nb_ref[...],
            cos_ref[...], sin_ref[...], eps=eps,
        )
        lane = lax.broadcasted_iota(jnp.int32, k.shape, 1)
        values, exp = _ue8m0_quantize(k)
        # Record lanes [0, 128): key bytes; [128, 256): scale byte, then zeros.
        # Kept as two 128-lane halves: Mosaic strided loads need a 128-wide ref.
        key_rec_ref[...] = lax.bitcast_convert_type(values, jnp.uint8).astype(jnp.uint32)
        tail_rec_ref[...] = jnp.where(lane == 0, (exp + 127).astype(jnp.uint32), jnp.uint32(0))

        # A cache row holds tokens 4r .. 4r + 3; token 4r + b is byte b of each
        # 32-bit word, as dsa_key_store merges them.
        halves = []
        for ref in (key_rec_ref, tail_rec_ref):
            w = ref[pl.ds(0, rows_per_page, stride=_TOKENS_PER_ROW), :]
            for b in range(1, _TOKENS_PER_ROW):
                part = ref[pl.ds(b, rows_per_page, stride=_TOKENS_PER_ROW), :]
                w = w | (part << jnp.uint32(8 * b))
            halves.append(w)
        words = jnp.concatenate(halves, axis=1)
        for r in range(rows_per_page):
            page_buf[slot, r] = pltpu.bitcast(words[r : r + 1], jnp.uint8)
        _write(slot, page).start()
        pending[slot] = 1

    @pl.when(i == pl.num_programs(0) - 1)
    def _():
        for s in range(2):

            @pl.when(pending[s] == 1)
            def _():
                _write(s, 0).wait()
                pending[s] = 0


@functools.partial(jax.jit, static_argnames=("eps", "interpret"))
def key_prologue_pallas(
    hidden: jax.Array,  # bf16[T, hidden]; T a multiple of page_size
    wk: jax.Array,  # f8e4m3fn[hidden, 128]; the linear's [K, N] weight
    wk_scale: jax.Array,  # f32[128]; per output channel
    norm_weight: jax.Array,  # f32[128]
    norm_bias: jax.Array,  # f32[128]
    rope_cos: jax.Array,  # bf16[T, rope_dim // 2]
    rope_sin: jax.Array,  # bf16[T, rope_dim // 2]
    cache_k: jax.Array,  # u8[P, page_size // 4, 4, 256]
    targets: jax.Array,  # i32[T // page_size]; from key_prologue_targets
    *,
    eps: float,
    interpret: bool = False,
) -> jax.Array:
    """Compute and store the index keys of whole token pages; returns the cache."""
    num_tokens, hidden_dim = hidden.shape
    num_pages, rows_per_page, tokens_per_row, record_bytes = cache_k.shape
    page_size = rows_per_page * tokens_per_row
    if (tokens_per_row, record_bytes) != (_TOKENS_PER_ROW, _RECORD_BYTES):
        raise ValueError(f"unexpected index-key cache shape {cache_k.shape}")
    if wk.shape != (hidden_dim, _HEAD_DIM) or wk.dtype != jnp.float8_e4m3fn:
        raise ValueError(f"wk must be f8e4m3fn[{hidden_dim}, {_HEAD_DIM}]")
    if num_tokens % page_size or targets.shape != (num_tokens // page_size,):
        raise ValueError("tokens must fill whole pages, one target per page")
    half = rope_cos.shape[1]
    if rope_cos.shape != (num_tokens, half) or 2 * half > _HEAD_DIM:
        raise ValueError(f"bad rope table shape {rope_cos.shape}")

    # Expand the rope tables to one value per lane: cos/sin repeat for each
    # pair, then identity past rope_dim.
    pad = ((0, 0), (0, _HEAD_DIM - 2 * half))
    cos = jnp.pad(jnp.repeat(rope_cos.astype(jnp.bfloat16), 2, axis=1), pad, constant_values=1)
    sin = jnp.pad(jnp.repeat(rope_sin.astype(jnp.bfloat16), 2, axis=1), pad)

    row = lambda a: a.astype(jnp.float32).reshape(1, _HEAD_DIM)
    whole = lambda shape: pl.BlockSpec(shape, lambda i, t: (0, 0))
    tokens = lambda width: pl.BlockSpec((page_size, width), lambda i, t: (i, 0))
    return pl.pallas_call(
        functools.partial(_kernel, eps=eps),
        out_shape=jax.ShapeDtypeStruct(cache_k.shape, cache_k.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_tokens // page_size,),
            in_specs=[
                tokens(hidden_dim),
                whole((hidden_dim, _HEAD_DIM)),
                whole((1, _HEAD_DIM)),
                whole((1, _HEAD_DIM)),
                whole((1, _HEAD_DIM)),
                tokens(_HEAD_DIM),
                tokens(_HEAD_DIM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=[
                pltpu.VMEM((page_size, _HEAD_DIM), jnp.uint32),
                pltpu.VMEM((page_size, _HEAD_DIM), jnp.uint32),
                pltpu.VMEM((2, rows_per_page, tokens_per_row, record_bytes), jnp.uint8),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SMEM((2,), jnp.int32),
            ],
        ),
        # Pages are written in order through a two-slot buffer.
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        input_output_aliases={8: 0},
        interpret=interpret,
        name="dsa_index_key_prologue",
    )(
        targets.astype(jnp.int32),
        hidden,
        wk,
        row(wk_scale),
        row(norm_weight),
        row(norm_bias),
        cos,
        sin,
        cache_k,
    )
