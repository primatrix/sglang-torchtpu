"""Test only: where do the ~2.85 ms of extend attention go once gather is gone?

bench_pregathered_ab (exp-rbyhjr88kr) found the #2246 extend_c1 step at ~2.85 ms
with every chunk's KV gathered in advance (online: 3.62-3.90 ms). This script
runs that pre-gathered call (random top-k, #2246 config: SparseCore layout,
chunk 128 / batch 16), takes one xprof trace of 5 calls and reads the TensorCore
"XLA Ops" timeline of each call:

* ``PGP_CALL``: call span, TC op-line busy / idle, time per op class (attention
  kernel ``MLA-*``, q pad, output slice, copies, other) by self time;
* ``PGP_KERNEL``: the 16 attention kernel events: count, duration mean / min /
  max, gaps between consecutive kernels;
* ``PGP_TOP``: the 12 largest ops by self time;
* ``PGP_ROOF``: the kernel's FLOPs and KV bytes per call, and what they imply at
  the measured kernel time (achieved TFLOP/s and GB/s).
"""

import collections
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
CHUNK, TOPK = 128, 2048
CALLS = 5


def main():
    xp = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pgp_xprof"
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    from xplane_summary import _planes
    from sglang_tpu.attention.pallas.mla_sparse import kv_cache_utils as kcu
    from sglang_tpu.attention.pallas.mla_sparse import native_sc_gather as nsc
    from sglang_tpu.attention.pallas.mla_sparse.kernel import sparse_ragged_paged_attention

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
    kw = dict(sm_scale=sm, k_scale=1.0, cache_layout="sparsecore",
              gather_and_attention_chunk_size=CHUNK, attention_kernel_batch_size=16,
              sparsecore_atoms_per_batch=16)
    cu, dist = jnp.array([0, T], jnp.int32), jnp.array([0, 0, 1], jnp.int32)
    pre = jax.jit(lambda q_, rows, gn, gr: sparse_ragged_paged_attention(
        q_, nope, rope, rows, None, cu, dist, pregathered=(gn, gr), **kw)[..., :512])

    @jax.jit
    def prepare(rows):
        gn, gr = [], []
        for ci in range(T // CHUNK):
            a, b = nsc.dsa_gather_native_sc(nope, rope, rows[ci * CHUNK:(ci + 1) * CHUNK],
                                            out_size=CHUNK * TOPK, atoms_per_batch=16)
            gn.append(a.reshape(CHUNK, TOPK, 128))
            gr.append(b.reshape(CHUNK, TOPK // nsc.ROPE_PACKING, 128))
        return tuple(gn), tuple(gr)

    rows = jnp.asarray(phys(np.asarray(c["topk"]).astype(np.int32)))
    gn, gr = prepare(rows)
    for _ in range(3):
        o = pre(q, rows, gn, gr)
    jax.block_until_ready(o)
    jax.profiler.start_trace(xp)
    for _ in range(CALLS):
        o = pre(q, rows, gn, gr)
    jax.block_until_ready(o)
    jax.profiler.stop_trace()

    lines = []
    for path in glob.glob(os.path.join(xp, "**", "*.xplane.pb"), recursive=True):
        for pname, names, plines in _planes(memoryview(open(path, "rb").read())):
            if pname.startswith("/device:TPU") and "SparseCore" not in pname:
                for lname, evs in plines:
                    lines.append((pname, lname, [(names.get(m, "?"), o_, o_ + d)
                                                 for m, o_, d in evs]))
    plane = min(p for p, l, ev in lines if l == "XLA Modules" and ev)
    mods = sorted((s, e) for p, l, ev in lines if p == plane and l == "XLA Modules"
                  for _, s, e in ev if e - s > 1e8)  # > 100 us
    ops = sorted((x for p, l, ev in lines if p == plane and l == "XLA Ops" for x in ev),
                 key=lambda x: (x[1], -(x[2] - x[1])))
    us = lambda ps: round(ps / 1e6, 1)

    def klass(nm):
        if "MLA-" in nm:
            return "attn_kernel"
        if "pad" in nm and "q_hbm" in nm:
            return "q_pad"
        if nm.startswith("slice") or "%slice" in nm:
            return "out_slice"
        if "copy" in nm:
            return "copy"
        return "other"

    top = collections.Counter()
    kdur = []
    for ci, (w0, w1) in enumerate(mods[:CALLS]):
        evs = [(nm, max(s, w0), min(e, w1)) for nm, s, e in ops if e > w0 and s < w1]
        self_t, stack = collections.Counter(), []
        for nm, s, e in evs:
            while stack and stack[-1][2] <= s:
                stack.pop()
            if stack:
                self_t[stack[-1][0]] -= min(e, stack[-1][2]) - s
            self_t[nm] += e - s
            stack.append((nm, s, e))
        busy, end = 0, None
        for _, s, e in evs:
            if end is None or s > end:
                busy += e - s
                end = e
            elif e > end:
                busy += e - end
                end = e
        cls = collections.Counter()
        for nm, t in self_t.items():
            cls[klass(nm)] += t
            top[nm[:70]] += t
        ks = sorted((s, e) for nm, s, e in evs if "MLA-" in nm and
                    not any(nm == p for p, _, _ in []))
        # Top-level attention kernel events only (no nested children).
        ks = [(s, e) for s, e in ks if not any(s2 < s and e <= e2 for s2, e2 in ks)]
        gaps = [ks[i + 1][0] - ks[i][1] for i in range(len(ks) - 1)]
        kdur += [e - s for s, e in ks]
        print("PGP_CALL " + json.dumps({
            "call": ci, "span_us": us(w1 - w0), "tc_busy_us": us(busy),
            "tc_idle_us": us(w1 - w0 - busy),
            "classes_us": {k: us(v) for k, v in cls.most_common()},
            "kernels": len(ks), "first_kernel_at_us": us(ks[0][0] - w0) if ks else None,
            "last_kernel_end_to_call_end_us": us(w1 - ks[-1][1]) if ks else None,
            "gap_sum_us": us(sum(gaps)), "gap_max_us": us(max(gaps)) if gaps else None,
        }), flush=True)
    if kdur:
        print("PGP_KERNEL " + json.dumps({
            "n": len(kdur), "mean_us": us(float(np.mean(kdur))), "min_us": us(min(kdur)),
            "max_us": us(max(kdur)), "sum_per_call_us": us(sum(kdur) / max(1, len(mods[:CALLS])))
        }), flush=True)
    calls = max(1, len(mods[:CALLS]))
    print("PGP_TOP " + json.dumps([(k, us(v / calls)) for k, v in top.most_common(12)]),
          flush=True)
    # Per call: QK over 576 dims + PV over 512 dims, 2 FLOP per MAC.
    flops = 2.0 * T * H * TOPK * (576 + 512)
    kv_bytes = T * TOPK * (512 + 128)  # NoPE 512 B + RoPE slot 128 B per gathered row
    if kdur:
        k_s = sum(kdur) / calls / 1e12
        print("PGP_ROOF " + json.dumps({
            "flops_per_call": flops, "kv_bytes_per_call": kv_bytes,
            "kernel_s_per_call": round(k_s, 6),
            "achieved_tflops": round(flops / k_s / 1e12, 1),
            "achieved_kv_gbps": round(kv_bytes / k_s / 1e9, 1)}), flush=True)
    print("PGP_DONE", flush=True)


if __name__ == "__main__":
    main()
