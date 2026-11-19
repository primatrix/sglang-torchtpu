"""Test only: Source vs Target DSA attention on one footing (physical slots, JAX).

extend_c1 (2048 queries, 49K prefix, top-k 2048, page 64, FP8 KV), single chip,
both called directly from JAX in one process (no torch op, so #2070 does not
apply). Every row uses physical cache slots.

* ``src_online``: Source fp8-qk-pv, split pipeline, its resolved tiles
  (``_run_exact`` body with ``PhysicalTopKValidation``; ported modules, which
  are bit-identical to sglang-jax-private c948bdd). Includes the KV write.
* ``src_gather``: Source SparseCore gathers alone, all microbatches.
* ``src_pre``: Source TensorCore kernel alone on pre-gathered microbatches.
* ``tgt_online``: Target per-query kernel, TensorCore layout, its default chunk,
  ``topk_is_physical=True``. Includes the KV write.
* ``tgt_gather`` / ``tgt_pre``: the same split for Target.

Each row also gets one absolute TC / SparseCore timeline (xplane_timeline).
Prints ``SVT`` lines.
"""

import argparse
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "tpu")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_fp8_attention_ab as att  # noqa: E402
from check_physical_topk import to_physical  # noqa: E402

FP = att.FP = 64
CASE = "pr2198_extend_c1"


def emit(**kw):
    print("SVT " + json.dumps(kw), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--xprof-dir", default="/tmp/svt")
    p.add_argument("--src-tiles", default="", help="override bq_sparse,bq,b_topk")
    a = p.parse_args()

    import jax
    import jax.numpy as jnp

    from sglang_tpu.attention.pallas import mla_sparse_fused as mf
    from sglang_tpu.attention.pallas.mla_sparse import dsa_gather, kv_cache_utils as kcu
    from sglang_tpu.attention.pallas.mla_sparse.kernel import (
        DEFAULT_GATHER_AND_ATTENTION_CHUNK_SIZE as TCHUNK,
        prepare_gather_indices,
        prepare_q_inputs,
        sparse_attention_chunk,
    )
    from sglang_tpu.attention.pallas.source_dsa.configs.dsa_config import DsaConfig
    from sglang_tpu.attention.pallas.source_dsa.kernels.dsa import exact_attention as ea
    from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.fp8_kv_cache import (
        SplitFP8KVCache, cache_primary, map_cache_parts,
    )
    from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.gather import _gather_split_fp8
    from sglang_tpu.attention.pallas.source_dsa.layers.attention import (
        dsa_sparse_backend_helpers as sh,
    )
    from xplane_timeline import call_timeline

    f8 = jnp.float8_e4m3fn
    c = att.case(CASE)
    P, T, S, K = c["P"], c["T"], c["S"], c["topk"].shape[1]
    n = P * FP
    kv0, pe0 = jnp.asarray(c["prefix_kv"]), jnp.asarray(c["prefix_pe"])
    ql, qr, kvc, kpe = (jnp.asarray(c[k]) for k in ("q_latent", "q_rope", "kv_c", "k_pe"))
    kvl, pi, cuq = (jnp.asarray(c[k]) for k in ("kv", "pi", "cu_q"))
    cukv = jnp.asarray((np.arange(S + 1) * c["ppr"] * FP).astype(np.int32))
    sm = float(c["sm_scale"])
    slots = jnp.asarray(to_physical(c["topk"], c["pi"], c["ppr"], c["cu_q"], page=FP))
    rows_bf = att.bf16_rows(c)

    def timed(call, sync):
        med, runs = att._median_time(call, sync, a.iters, a.repeats)
        return round(med, 1), [round(min(runs), 1), round(max(runs), 1)]

    def timeline(call, sync, tag, attn_key, gather_keys):
        d = os.path.join(a.xprof_dir, tag)
        sync()
        jax.profiler.start_trace(d)
        try:
            for _ in range(4):
                call()
            sync()
        finally:
            jax.profiler.stop_trace()
        r = call_timeline(d, call=1, attn_key=attn_key, gather_keys=gather_keys)
        keep = ("module_us", "attn_busy_us", "sc_busy_us", "attn_and_sc_us", "lead_us",
                "tail_us", "gaps_us", "sc_busy_in_gaps_us", "attn_without_sc_us",
                "attn_chunks", "sc_busy", "tc_other_top", "error")
        return {k: r[k] for k in keep if k in r}

    def run(tag, fn, args, attn_key, gather_keys, check=None):
        r = dict(variant=tag)
        try:
            jf = jax.jit(fn)
            out = jf(*args)
            sync = lambda: jax.block_until_ready(out)  # noqa: E731
            sync()
            if check is not None:
                r["accuracy_vs_bf16"] = att.error_vs_bf16(
                    c, np.asarray(check(out)).astype(np.float64), rows_bf)
            r["us"], r["us_range"] = timed(lambda: jf(*args), lambda: jax.block_until_ready(jf(*args)))
            r["timeline"] = timeline(lambda: jf(*args), sync, tag, attn_key, gather_keys)
        except Exception as e:  # report, do not hide; the other rows still run
            import traceback

            traceback.print_exc()
            r["fail"] = repr(e)[:1500]
        emit(**r)

    # ---------------- Source ----------------
    empty = SplitFP8KVCache(jnp.zeros((P, FP, 4, 128), jnp.uint8),
                            jnp.zeros((P, FP // 4, 4, 128), jnp.uint8))
    filled = jax.jit(lambda cc, kv_, pe_: sh._scatter_fused_kv_paged(
        cc, kv_, pe_, jnp.array([n], jnp.int32), jnp.arange(P, dtype=jnp.int32),
        jnp.array([0, n], jnp.int32), jnp.array([0, n], jnp.int32),
        kv_lora_rank=att.LAT, fp8_dtype=f8))(empty, kv0, pe0)
    plan = DsaConfig(precision="fp8-qk-pv").resolve(T, eligible=True, is_prefill=True, fp8_kv=True)
    tiles = tuple(int(x) for x in a.src_tiles.split(",")) if a.src_tiles else plan.tiles
    bqs, bq, bk = tiles
    emit(stage="setup", T=T, topk=K, pages=P, src_plan=str(plan), src_tiles=tiles,
         tgt_chunk=TCHUNK)

    def src_step(lat, rope):
        cache4d = sh._scatter_fused_kv_paged(
            SplitFP8KVCache(lat, rope), kvc, kpe, kvl, pi, cuq, cukv,
            kv_lora_rank=att.LAT, fp8_dtype=f8)
        valid = (slots >= 0) & (slots < cache_primary(cache4d).shape[0] * FP)
        phys = jnp.where(valid, slots, jnp.int32(0))
        counts = jnp.sum(valid, axis=1, dtype=jnp.int32)
        cache = map_cache_parts(lambda x: x.reshape(-1, *x.shape[2:]), cache4d)
        pad = sh._pad_first_axis
        out = ea.sparse_core_tensor_core_dsa(
            pad(ql, bqs), pad(qr, bqs), cache, pad(phys, bqs), pad(counts, bqs), sm,
            bq_sparse=bqs, bq=bq, b_topk=bk, fp8_dtype=f8, fp8_q=True, fp8_pv=True,
            split_gather_buffers=True)
        return out[:T]

    src_cache = map_cache_parts(lambda x: x.reshape(-1, *x.shape[2:]), filled)
    counts = jnp.full((T,), K, jnp.int32)
    nmb = T // bqs

    def src_gather(cache):
        return tuple(_gather_split_fp8(cache, slots[i * bqs:(i + 1) * bqs].reshape(-1))
                     for i in range(nmb))

    def src_pre(*groups):
        outs = []
        for i, g in enumerate(groups):
            cur = map_cache_parts(lambda x: x.reshape(1, bqs, K, *x.shape[1:]), g)
            outs.append(ea._tensor_core_attention(
                ql[i * bqs:(i + 1) * bqs], qr[i * bqs:(i + 1) * bqs], cur,
                counts[i * bqs:(i + 1) * bqs], jnp.int32(0), sm_scale=sm, bq=bq, b_topk=bk,
                fp8_dtype=f8, fp8_q=True, fp8_pv=True))
        return jnp.concatenate(outs, axis=0)

    s_attn, s_gat = "dsa_tensor_core_attention", ("dsa_sparse_core_gather",)
    run("src_online", src_step, (filled.latent, filled.rope), s_attn, s_gat, check=lambda o: o)
    run("src_gather", src_gather, (src_cache,), s_attn, s_gat)
    groups = jax.block_until_ready(jax.jit(src_gather)(src_cache))
    run("src_pre", src_pre, groups, s_attn, s_gat, check=lambda o: o)
    del groups

    # ---------------- Target ----------------
    fill, _ = att.target_fns(c, qblock=False)
    tn, tr = jax.jit(fill, static_argnums=(4,))(
        jnp.zeros((P, FP, 4, 128), jnp.uint8), jnp.zeros((P, FP // 4, 4, 128), jnp.uint8),
        kv0, pe0, n, jnp.arange(P, dtype=jnp.int32))
    dist = jnp.asarray(np.array([0, 0, S], np.int32))
    fmax = float(jnp.finfo(f8).max)
    q8 = lambda x: jnp.clip(x.astype(jnp.float32), -fmax, fmax).astype(f8)  # noqa: E731

    def tgt_step(no, ro):
        out, _, _ = mf.sparse_mla_attention_fused(
            ql, qr, q8(kvc), q8(kpe), no, ro, kvl, pi, cuq, dist, slots,
            sm_scale=sm, k_scale=1.0, topk_is_physical=True)
        return out

    kv_lens, phys_t = prepare_gather_indices(slots, None, cuq, page_size=FP, total_num_pages=P)
    qfull = prepare_q_inputs(jnp.concatenate([ql, qr], axis=-1))
    nch = T // TCHUNK

    def tgt_gather(no, ro):
        tl = []
        for i in range(nch):
            gn, gr = dsa_gather.dsa_gather(
                no, ro, phys_t[i * TCHUNK:(i + 1) * TCHUNK].reshape(-1),
                num_valid_indices=jnp.int32(TCHUNK * K))
            tl.append((gn.reshape(TCHUNK, K * kcu.WORD_BYTES, kcu.TILE_LANE_BYTES),
                       gr.reshape(TCHUNK, K, -1)))
        return tuple(tl)

    def tgt_pre(no, ro, *tl):
        q = qfull
        for i in range(nch):
            q = sparse_attention_chunk(
                q, no, ro, kv_lens, phys_t, None, start_pos=i * TCHUNK,
                end_pos=(i + 1) * TCHUNK, token_end=jnp.int32(T), sm_scale=sm, k_scale=1.0,
                topk_override=K, gathered_group=tl[i])
        return q[:, :ql.shape[1], :att.LAT]

    t_attn, t_gat = "MLA-", ("sc_dsa_gather",)
    run("tgt_online", tgt_step, (tn, tr), t_attn, t_gat, check=lambda o: o)
    run("tgt_gather", tgt_gather, (tn, tr), t_attn, t_gat)
    tl = jax.block_until_ready(jax.jit(tgt_gather)(tn, tr))
    run("tgt_pre", tgt_pre, (tn, tr, *tl), t_attn, t_gat, check=lambda o: o)
    print("SVT_DONE", flush=True)


if __name__ == "__main__":
    main()
