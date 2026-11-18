"""Test only: fused shared + private sparse attention on the real GLM-5.2 top-k dump.

``bench_shared_private.py`` (exp-2sgevbvpt5) split each group's top-k into
shared and private rows with two unchanged ``sparse_ragged_paged_attention``
calls. It lost for two reasons: the shared size had to be static (the minimum
intersection over all groups, so most shared rows were not shared), and the
second call cost ~400 us (a second q pad, the LSE merge). This version fixes
both:

* Gather: per 128-query chunk, one flat row list. Per group of ``g`` queries:
  its shared segment (the intersection, its own length), then each query's
  private segment, every segment padded only to 64 rows. One
  ``dsa_gather_native_sc`` call per chunk with ``num_live_indices``, so the
  gather walks the live rows only.
* Attention: one Pallas TensorCore kernel, one grid step per group. It DMAs the
  group's shared window and each query's private window out of the flat buffer
  (offsets and lengths scalar-prefetched, double-buffered across groups),
  attends the shared rows once as ``g * H`` heads (MLA's latent K/V is shared
  across heads), attends each private window per query, and merges the two in
  registers (online softmax). One q pad, no separate merge.

Not measured: building the intersections. They are built on the host (numpy)
and passed in; the device cost of building them is a separate question.

Prints ``SPF`` lines (``SPFT``: short timing lines).
"""

import argparse
import functools
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_fp8_attention_ab as att  # noqa: E402

FP = att.FP = 64
CASE = "pr2198_extend_c1"
CHUNK, BATCH, TOPK = 128, 16, 2048
SEG = 64  # segment alignment (gather atom rows; keeps rope rows 16-aligned)
CAP = CHUNK * TOPK + 32768  # per-chunk flat capacity, a multiple of 32 owners * 16 atoms * 64


def build_layout(raw, phys, g, n_slots):
    """Host-side flat row lists. ``raw``: [T, 2048] logical rows (all valid)."""
    T = raw.shape[0]
    nb = CHUNK // g
    sets = [np.sort(x) for x in raw]
    nch = T // CHUNK
    idx = np.zeros((nch, CAP), np.int32)
    live = np.zeros((nch,), np.int32)
    s_off = np.zeros((nch, nb), np.int32)
    ks = np.zeros((nch, nb), np.int32)
    p_off = np.zeros((nch, CHUNK), np.int32)
    kp = np.zeros((nch, CHUNK), np.int32)
    pad_seed = [0]

    def seg(buf, off, rows):
        n = len(rows)
        npad = -(-max(n, 1) // SEG) * SEG
        # Padding reads spread over the cache (masked in the kernel).
        pad = (np.arange(npad - n, dtype=np.int64) * 104729 + pad_seed[0]) % n_slots
        pad_seed[0] += 7919
        buf[off:off + n] = phys(rows)
        buf[off + n:off + npad] = pad
        return off + npad

    for c in range(nch):
        off = 0
        for b in range(nb):
            q0 = c * CHUNK + b * g
            inter = sets[q0]
            for t in sets[q0 + 1:q0 + g]:
                inter = np.intersect1d(inter, t, assume_unique=True)
            s_off[c, b], ks[c, b] = off, len(inter)
            off = seg(idx[c], off, inter)
            for i in range(g):
                pv = np.setdiff1d(sets[q0 + i], inter, assume_unique=True)
                p_off[c, b * g + i], kp[c, b * g + i] = off, len(pv)
                off = seg(idx[c], off, pv)
        live[c] = off
    s_w = max(SEG, -(-int(ks.max()) // SEG) * SEG)
    p_w = max(SEG, -(-int(kp.max()) // SEG) * SEG)
    assert int(live.max()) + max(s_w, p_w) <= CAP, (live.max(), s_w, p_w)
    return dict(idx=idx, live=live, s_off=s_off, ks=ks, p_off=p_off, kp=kp,
                s_w=s_w, p_w=p_w, rows_frac=float(live.sum()) / (T * TOPK),
                ks_mean=float(ks.mean()), ks_min=int(ks.min()))


def make_fused(g, s_w, p_w, sm_scale, num_heads, head_dim):
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    nb = CHUNK // g
    f8 = jnp.float8_e4m3fn

    def unpack(nope_u32, rope_u32, w, n):
        # As kernel.unpack_bkv (SparseCore layout).
        nope = pltpu.bitcast(pltpu.bitcast(nope_u32, jnp.uint8).reshape(w, 512), f8)
        rope = pltpu.bitcast(pltpu.bitcast(rope_u32, jnp.uint8).reshape(w, 128), f8)
        kv = jnp.concatenate([nope, rope], axis=-1)
        if kv.shape[-1] < head_dim:
            kv = jnp.pad(kv, ((0, 0), (0, head_dim - kv.shape[-1])))
        span = lax.broadcasted_iota(jnp.int32, kv.shape, 0)
        return jnp.where(span < n, kv, 0)

    def partial_attn(q, kv, n):
        s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32) * sm_scale
        valid = lax.broadcasted_iota(jnp.int32, s.shape, 1) < n
        s = jnp.where(valid, s, jnp.finfo(jnp.float32).min)
        m = jnp.max(s, axis=1, keepdims=True)
        p = jnp.where(valid, jnp.exp(s - m), 0.0)
        l_sum = jnp.sum(p, axis=1, keepdims=True)
        acc = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)
        return m, l_sum, acc

    def body(s_off_ref, ks_ref, p_off_ref, kp_ref, q_ref, nope_hbm, rope_hbm, o_ref,
             sn, sr, pn, pr, sems):
        b = pl.program_id(0)
        slot = b % 2

        def copies(bb, sl):
            so = pl.multiple_of(s_off_ref[bb], SEG)
            cps = [
                pltpu.make_async_copy(nope_hbm.at[pl.ds(so, s_w)], sn.at[sl], sems.at[0, sl]),
                pltpu.make_async_copy(rope_hbm.at[pl.ds(so // 4, s_w // 4)], sr.at[sl],
                                      sems.at[1, sl]),
            ]
            for i in range(g):
                po = pl.multiple_of(p_off_ref[bb * g + i], SEG)
                cps.append(pltpu.make_async_copy(nope_hbm.at[pl.ds(po, p_w)], pn.at[sl, i],
                                                 sems.at[2 + 2 * i, sl]))
                cps.append(pltpu.make_async_copy(rope_hbm.at[pl.ds(po // 4, p_w // 4)],
                                                 pr.at[sl, i], sems.at[3 + 2 * i, sl]))
            return cps

        @pl.when(b == 0)
        def _():
            for cp in copies(0, 0):
                cp.start()

        @pl.when(b + 1 < nb)
        def _():
            for cp in copies(b + 1, 1 - slot):
                cp.start()

        for cp in copies(b, slot):
            cp.wait()

        q = q_ref[...]  # [g, H, D]
        kvs = unpack(sn[slot], sr[slot], s_w, ks_ref[b])
        m1, l1, a1 = partial_attn(q.reshape(g * num_heads, head_dim), kvs, ks_ref[b])
        for i in range(g):
            kvp = unpack(pn[slot, i], pr[slot, i], p_w, kp_ref[b * g + i])
            m2, l2, a2 = partial_attn(q[i], kvp, kp_ref[b * g + i])
            r = slice(i * num_heads, (i + 1) * num_heads)
            m = jnp.maximum(m1[r], m2)
            e1, e2 = jnp.exp(m1[r] - m), jnp.exp(m2 - m)
            den = l1[r] * e1 + l2 * e2
            out = (a1[r] * e1 + a2 * e2) * pltpu.reciprocal(den, approx=True)
            o_ref[i] = out.astype(o_ref.dtype)

    def chunk_call(c, q, nope_g, rope_g, s_off, ks, p_off, kp):
        base = c * nb
        blk = pl.BlockSpec((g, num_heads, head_dim), lambda b, *_: (base + b, 0, 0))
        return pl.pallas_call(
            body,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=4,
                grid=(nb,),
                in_specs=[blk, pl.BlockSpec(memory_space=pltpu.HBM),
                          pl.BlockSpec(memory_space=pltpu.HBM)],
                out_specs=blk,
                scratch_shapes=[
                    pltpu.VMEM((2, s_w, 128), jnp.uint32),
                    pltpu.VMEM((2, s_w // 4, 128), jnp.uint32),
                    pltpu.VMEM((2, g, p_w, 128), jnp.uint32),
                    pltpu.VMEM((2, g, p_w // 4, 128), jnp.uint32),
                    pltpu.SemaphoreType.DMA((2 + 2 * g, 2)),
                ],
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",), vmem_limit_bytes=100 * 1024 * 1024),
            out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
            input_output_aliases={4: 0},
            name=f"MLA-shared-private-g{g}-s{s_w}-p{p_w}",
        )(s_off, ks, p_off, kp, q, nope_g, rope_g)

    return chunk_call


def build_jobs(L, g, blk):
    """Variable-length schedule: each segment becomes cdiv(n, blk) jobs of ``blk`` rows.

    Per group, the shared segment's jobs come first, then each query's private
    jobs. A job past its segment's end reads the next segment's rows (or unwritten
    tail rows); the kernel masks rows >= ``nv``. Flags: 1 = first job of its
    segment, 2 = last. ``js`` is the chunk-local index of each group's first job,
    so the DMA slot (job index % 2) needs no state across grid steps.
    """
    nch, nb = L["ks"].shape
    jobs = [[[] for _ in range(nb)] for _ in range(nch)]
    for c in range(nch):
        for b in range(nb):
            segs = [(-1, L["s_off"][c, b], L["ks"][c, b])]
            segs += [(i, L["p_off"][c, b * g + i], L["kp"][c, b * g + i]) for i in range(g)]
            for kind, off, n in segs:
                nj = max(1, -(-int(n) // blk))
                for j in range(nj):
                    jobs[c][b].append((off + j * blk, int(np.clip(n - j * blk, 0, blk)), kind,
                                       (1 if j == 0 else 0) | (2 if j == nj - 1 else 0)))
    jmax = max(len(x) for ch in jobs for x in ch)
    arr = {k: np.zeros((nch, nb * jmax), np.int32) for k in ("off", "nv", "kd", "fl")}
    nj = np.zeros((nch, nb), np.int32)
    js = np.zeros((nch, nb), np.int32)
    for c in range(nch):
        acc = 0
        for b in range(nb):
            nj[c, b], js[c, b] = len(jobs[c][b]), acc
            acc += len(jobs[c][b])
            for j, (off, nv, kd, fl) in enumerate(jobs[c][b]):
                for k, v in zip(("off", "nv", "kd", "fl"), (off, nv, kd, fl)):
                    arr[k][c, b * jmax + j] = v
    assert int(arr["off"].max()) + blk <= CAP
    work = sum(len(x) for ch in jobs for x in ch) * blk
    return dict(nj=nj, js=js, jmax=jmax, work_frac=work / (L["ks"].shape[0] * CHUNK * TOPK),
                **arr)


def make_fused_var(g, blk, jmax, sm_scale, num_heads, head_dim):
    """One grid step per group; an in-kernel loop over its ``blk``-row jobs.

    KV blocks are double-buffered across jobs and across groups (the last job of
    group b prefetches group b+1's first). Online softmax: shared state for the
    group's ``g * H`` rows, private state for the current query; the last private
    job of query i merges the two and writes its output.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    nb = CHUNK // g
    f8 = jnp.float8_e4m3fn
    gh = g * num_heads
    neg = jnp.finfo(jnp.float32).min

    def unpack(nope_u32, rope_u32, n):
        nope = pltpu.bitcast(pltpu.bitcast(nope_u32, jnp.uint8).reshape(blk, 512), f8)
        rope = pltpu.bitcast(pltpu.bitcast(rope_u32, jnp.uint8).reshape(blk, 128), f8)
        kv = jnp.concatenate([nope, rope], axis=-1)
        if kv.shape[-1] < head_dim:
            kv = jnp.pad(kv, ((0, 0), (0, head_dim - kv.shape[-1])))
        # Rows past nv may be another segment's or unwritten (fp8 NaN): zero them.
        span = lax.broadcasted_iota(jnp.int32, kv.shape, 0)
        return jnp.where(span < n, kv, 0)

    def update(q, kv, n, m_ref, l_ref, a_ref):
        s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32) * sm_scale
        valid = lax.broadcasted_iota(jnp.int32, s.shape, 1) < n
        s = jnp.where(valid, s, neg)
        m_old = m_ref[...]
        m_new = jnp.maximum(m_old, jnp.max(s, axis=1, keepdims=True))
        alpha = jnp.exp(m_old - m_new)
        p = jnp.where(valid, jnp.exp(s - m_new), 0.0)
        l_ref[...] = alpha * l_ref[...] + jnp.sum(p, axis=1, keepdims=True)
        # Same operand dtypes as partial_attn above (f32 p, fp8 kv).
        a_ref[...] = alpha * a_ref[...] + jnp.einsum(
            "nm,md->nd", p, kv, preferred_element_type=jnp.float32)
        m_ref[...] = m_new

    def body(nj_ref, js_ref, off_ref, nv_ref, kd_ref, fl_ref, q_ref, nope_hbm, rope_hbm,
             o_ref, bn, br, ms, ls, as_, mp, lp, ap, sems):
        b = pl.program_id(0)

        def dma(bb, j, sl):
            o = pl.multiple_of(off_ref[bb * jmax + j], SEG)
            return [
                pltpu.make_async_copy(nope_hbm.at[pl.ds(o, blk)], bn.at[sl], sems.at[0, sl]),
                pltpu.make_async_copy(rope_hbm.at[pl.ds(o // 4, blk // 4)], br.at[sl],
                                      sems.at[1, sl]),
            ]

        @pl.when(b == 0)
        def _():
            for cp in dma(0, 0, 0):
                cp.start()

        n_jobs = nj_ref[b]

        def step(j, carry):
            sl = (js_ref[b] + j) % 2

            @pl.when(j + 1 < n_jobs)
            def _():
                for cp in dma(b, j + 1, 1 - sl):
                    cp.start()

            @pl.when((j + 1 == n_jobs) & (b + 1 < nb))
            def _():
                for cp in dma(b + 1, 0, 1 - sl):
                    cp.start()

            for cp in dma(b, j, sl):
                cp.wait()
            k = b * jmax + j
            n, kind, fl = nv_ref[k], kd_ref[k], fl_ref[k]
            kv = unpack(bn[sl], br[sl], n)

            @pl.when(kind < 0)
            def _():
                @pl.when((fl & 1) != 0)
                def _():
                    ms[...] = jnp.full(ms.shape, neg, jnp.float32)
                    ls[...] = jnp.zeros(ls.shape, jnp.float32)
                    as_[...] = jnp.zeros(as_.shape, jnp.float32)

                update(q_ref[...].reshape(gh, head_dim), kv, n, ms, ls, as_)

            @pl.when(kind >= 0)
            def _():
                @pl.when((fl & 1) != 0)
                def _():
                    mp[...] = jnp.full(mp.shape, neg, jnp.float32)
                    lp[...] = jnp.zeros(lp.shape, jnp.float32)
                    ap[...] = jnp.zeros(ap.shape, jnp.float32)

                update(q_ref[kind], kv, n, mp, lp, ap)

                @pl.when((fl & 2) != 0)
                def _():
                    r = pl.ds(pl.multiple_of(kind * num_heads, num_heads), num_heads)
                    m1, m2 = ms[r], mp[...]
                    m = jnp.maximum(m1, m2)
                    e1, e2 = jnp.exp(m1 - m), jnp.exp(m2 - m)
                    den = ls[r] * e1 + lp[...] * e2
                    out = (as_[r] * e1 + ap[...] * e2) * pltpu.reciprocal(den, approx=True)
                    o_ref[kind] = out.astype(o_ref.dtype)

            return carry

        lax.fori_loop(0, n_jobs, step, 0)

    def chunk_call(c, q, nope_g, rope_g, nj, js, off, nv, kd, fl):
        base = c * nb
        qb = pl.BlockSpec((g, num_heads, head_dim), lambda b, *_: (base + b, 0, 0))
        return pl.pallas_call(
            body,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=6,
                grid=(nb,),
                in_specs=[qb, pl.BlockSpec(memory_space=pltpu.HBM),
                          pl.BlockSpec(memory_space=pltpu.HBM)],
                out_specs=qb,
                scratch_shapes=[
                    pltpu.VMEM((2, blk, 128), jnp.uint32),
                    pltpu.VMEM((2, blk // 4, 128), jnp.uint32),
                    pltpu.VMEM((gh, 1), jnp.float32),
                    pltpu.VMEM((gh, 1), jnp.float32),
                    pltpu.VMEM((gh, head_dim), jnp.float32),
                    pltpu.VMEM((num_heads, 1), jnp.float32),
                    pltpu.VMEM((num_heads, 1), jnp.float32),
                    pltpu.VMEM((num_heads, head_dim), jnp.float32),
                    pltpu.SemaphoreType.DMA((2, 2)),
                ],
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",), vmem_limit_bytes=100 * 1024 * 1024),
            out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
            input_output_aliases={6: 0},
            name=f"MLA-shared-private-var-g{g}-b{blk}",
        )(nj, js, off, nv, kd, fl, q, nope_g, rope_g)

    return chunk_call


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--files", default="raw-0096-layer0,raw-0097-layer10,raw-0099-layer40")
    p.add_argument("--groups", default="4,8")
    # Variable-length private DMA: job block rows (multiples of SEG); 0 = the fixed kernel.
    p.add_argument("--blocks", default="0,128,256")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--xprof-dir", required=True)
    a = p.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    from xplane_summary import summarize
    from sglang_tpu.attention.pallas.mla_sparse import kv_cache_utils as kcu
    from sglang_tpu.attention.pallas.mla_sparse import native_sc_gather
    from sglang_tpu.attention.pallas.mla_sparse.kernel import (
        prepare_q_inputs,
        sparse_ragged_paged_attention,
    )

    f8 = jnp.float8_e4m3fn
    fmax = float(jnp.finfo(f8).max)
    q8 = lambda x: jnp.clip(x.astype(jnp.float32), -fmax, fmax).astype(f8)
    lay = kcu.KVCacheLayout.SPARSECORE
    c = att.case(CASE)
    P, T = c["P"], c["T"]
    n = P * FP
    ns = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.NOPE, lay, P, FP, 512)
    rs = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.ROPE, lay, P, FP, 64)
    fill = jax.jit(lambda nope, rope, kv_, pe_: kcu.update_sparse_mla_kv_cache_jax(
        nope, rope, kv_, pe_, jnp.array([n], jnp.int32), jnp.arange(P, dtype=jnp.int32),
        jnp.array([0, n], jnp.int32), nope_spec=ns, rope_spec=rs))
    nope, rope = fill(jnp.zeros(ns.shape, ns.jax_dtype), jnp.zeros(rs.shape, rs.jax_dtype),
                      q8(jnp.asarray(c["prefix_kv"])), q8(jnp.asarray(c["prefix_pe"])))
    q = jnp.concatenate([jnp.asarray(c["q_latent"]), jnp.asarray(c["q_rope"])], axis=-1)
    H = q.shape[1]
    sm = float(c["sm_scale"])
    pi = np.asarray(c["pi"])
    phys = lambda lg: np.where(lg >= 0, pi[np.maximum(lg, 0) // FP] * FP
                               + np.maximum(lg, 0) % FP, -1).astype(np.int32)

    @jax.jit
    def base_fn(q_, nope_, rope_, rows):
        return sparse_ragged_paged_attention(
            q_, nope_, rope_, rows, None, jnp.array([0, T], jnp.int32),
            jnp.array([0, 0, 1], jnp.int32), sm_scale=sm, k_scale=1.0,
            cache_layout="sparsecore", gather_and_attention_chunk_size=CHUNK,
            attention_kernel_batch_size=BATCH, sparsecore_atoms_per_batch=16)[..., :512]

    def timed(tag, fn):
        st = {}
        sync = lambda: jax.block_until_ready(st["o"])

        def call():
            st["o"] = fn()
            return st["o"]

        med, runs = att._median_time(call, sync, a.iters, a.repeats)
        d = f"{a.xprof_dir}/{tag}"
        jax.profiler.start_trace(d)
        for _ in range(a.iters):
            call()
        sync()
        jax.profiler.stop_trace()
        x = summarize(d, a.iters, top=6)
        return dict(us=round(med, 1), us_range=[round(min(runs), 1), round(max(runs), 1)],
                    module_us=round(sum(u for _, u in x.get("modules", [])), 1),
                    top_ops=[(nm[:40], round(u, 1)) for nm, u in x.get("ops", [])[:6]])

    paths = {os.path.basename(f)[:-4]: f
             for f in glob.glob(f"{a.raw_dir}/**/raw-*.npy", recursive=True)}
    for name in a.files.split(","):
        try:
            raw = np.load(paths[name]).astype(np.int32)
            assert raw.shape == (T, TOPK) and (raw >= 0).all(), raw.shape
            c["topk"] = raw
            rows_bf = att.bf16_rows(c)
            full = jnp.asarray(phys(raw))
            ref = np.asarray(base_fn(q, nope, rope, full)).astype(np.float32)
            row = {"file": name, "base_err": att.error_vs_bf16(c, ref.astype(np.float64), rows_bf)}
            row["base"] = timed(f"{name}_base", lambda: base_fn(q, nope, rope, full))
        except Exception as e:  # report, do not hide
            import traceback

            traceback.print_exc()
            print("SPF " + json.dumps({"file": name, "fail": repr(e)[:500]}), flush=True)
            continue
        print("SPF " + json.dumps(row), flush=True)
        print("SPFT " + json.dumps({"file": name, "base_us": row["base"]["us"],
                                    "base_module_us": row["base"]["module_us"]}), flush=True)
        for g, blk in ((int(x), int(y)) for x in a.groups.split(",")
                       for y in a.blocks.split(",")):
            r = {"file": name, "g": g, "blk": blk}
            try:
                L = build_layout(raw, phys, g, n)
                r.update(rows_frac=round(L["rows_frac"], 4), s_w=L["s_w"], p_w=L["p_w"],
                         ks_mean=round(L["ks_mean"], 1), ks_min=L["ks_min"])
                dev = {k: jnp.asarray(L[k]) for k in ("idx", "live", "s_off", "ks", "p_off", "kp")}
                if blk:
                    J = build_jobs(L, g, blk)
                    r.update(work_frac=round(J["work_frac"], 4), jmax=J["jmax"])
                    for k in ("nj", "js", "off", "nv", "kd", "fl"):
                        dev["j_" + k] = jnp.asarray(J[k])
                    call_chunk = make_fused_var(g, blk, J["jmax"], sm, H, 640)
                    chunk_args = lambda d, ci: [d["j_" + k][ci] for k in
                                                ("nj", "js", "off", "nv", "kd", "fl")]
                else:
                    call_chunk = make_fused(g, L["s_w"], L["p_w"], sm, H, 640)
                    chunk_args = lambda d, ci: [d[k][ci] for k in ("s_off", "ks", "p_off", "kp")]

                @jax.jit
                def fused(q_, nope_, rope_, d, call_chunk=call_chunk, chunk_args=chunk_args):
                    qp = prepare_q_inputs(q_)  # [T, H, 640]
                    for ci in range(T // CHUNK):
                        ng, rg = native_sc_gather.dsa_gather_native_sc(
                            nope_, rope_, d["idx"][ci].reshape(CAP // 64, 64), out_size=CAP,
                            atoms_per_batch=16, num_live_indices=d["live"][ci],
                            live_multiple=SEG)
                        qp = call_chunk(ci, qp, ng, rg, *chunk_args(d, ci))
                    return qp[:, :H, :512]

                out = np.asarray(fused(q, nope, rope, dev)).astype(np.float32)
                den = np.maximum(np.abs(ref), 1e-3)
                r["vs_base"] = dict(max_abs=float(np.abs(out - ref).max()),
                                    p99_rel=float(np.percentile(np.abs(out - ref) / den, 99)),
                                    finite=bool(np.isfinite(out).all()))
                r["err"] = att.error_vs_bf16(c, out.astype(np.float64), rows_bf)
                r.update(timed(f"{name}_fused_g{g}_b{blk}", lambda: fused(q, nope, rope, dev)))
            except Exception as e:  # report, do not hide
                import traceback

                traceback.print_exc()
                r["fail"] = repr(e)[:500]
            print("SPF " + json.dumps(r), flush=True)
            print("SPFT " + json.dumps({k: r.get(k) for k in (
                "file", "g", "blk", "rows_frac", "work_frac", "s_w", "p_w", "vs_base", "us",
                "module_us", "fail")}), flush=True)
    print("SPF_DONE", flush=True)


if __name__ == "__main__":
    main()
