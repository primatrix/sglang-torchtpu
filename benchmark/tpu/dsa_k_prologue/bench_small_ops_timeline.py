"""Test only: are the q pad, output slice and SC data-format ops serial or hidden?

Production-shaped extend_c1 step (as ``bench_q_layout_ab`` ``q_split``: row-major
``ql_nope`` / ``q_pe`` concatenated inside the jit, SparseCore layout, physical
slots, chunk 128 / batch 16 / atoms 16 / overlap_rope). One xprof trace; for the
second captured call, every event whose name matches a target (``pad``,
``slice``, ``sparse-core-data-format``, ``copy``, ``concatenate``) on any TPU
line is printed as a ``TGT`` line with:

* its start / duration relative to the call;
* ``sc_busy_us``: how much of it overlaps SparseCore gather work (union of
  ``native_sc`` events on all SparseCore lines);
* ``tc_other_us``: how much overlaps other TensorCore ops (attention etc.).

A target with ``sc_busy_us`` near its duration is hidden under the gather;
one near 0 is serial. ``SUM`` gives the totals and the call's serial time.
"""

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_fp8_attention_ab as att  # noqa: E402
from bench_chunk_timeline import inter, union  # noqa: E402
from check_physical_topk import to_physical  # noqa: E402

FP = att.FP = 64
CASE = "pr2198_extend_c1"
KW = dict(gather_and_attention_chunk_size=128, attention_kernel_batch_size=16,
          sparsecore_atoms_per_batch=16, sparsecore_gather_wait_mode="overlap_rope")
TARGETS = ("pad", "slice", "sparse-core-data-format", "copy", "concatenate")
GATHER = ("native_sc",)


def main():
    xp = sys.argv[1] if len(sys.argv) > 1 else "/tmp/small_ops"
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp
    from jax.experimental.layout import Format, Layout

    from xplane_summary import _planes
    from sglang_tpu.attention.pallas.mla_sparse import kv_cache_utils as kcu
    from sglang_tpu.attention.pallas.mla_sparse.kernel import sparse_ragged_paged_attention

    f8 = jnp.float8_e4m3fn
    fmax = float(jnp.finfo(f8).max)
    q8 = lambda x: jnp.clip(x.astype(jnp.float32), -fmax, fmax).astype(f8)
    lay = kcu.KVCacheLayout.SPARSECORE
    c = att.case(CASE)
    P, S = c["P"], c["S"]
    n = P * FP
    ns = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.NOPE, lay, P, FP, 512)
    rs = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.ROPE, lay, P, FP, 64)
    fill = jax.jit(lambda nope, rope, kv_, pe_: kcu.update_sparse_mla_kv_cache_jax(
        nope, rope, kv_, pe_, jnp.array([n], jnp.int32), jnp.arange(P, dtype=jnp.int32),
        jnp.array([0, n], jnp.int32), nope_spec=ns, rope_spec=rs))
    nope, rope = fill(jnp.zeros(ns.shape, ns.jax_dtype), jnp.zeros(rs.shape, rs.jax_dtype),
                      q8(jnp.asarray(c["prefix_kv"])), q8(jnp.asarray(c["prefix_pe"])))
    kvc, kpe = q8(jnp.asarray(c["kv_c"])), q8(jnp.asarray(c["k_pe"]))
    kvl, pi, cuq = (jnp.asarray(c[k]) for k in ("kv", "pi", "cu_q"))
    dist = jnp.asarray([0, 0, S], jnp.int32)
    rows = jnp.asarray(to_physical(c["topk"], c["pi"], c["ppr"], c["cu_q"], page=FP))
    sm = float(c["sm_scale"])

    def step(nope_, rope_, ql_, qpe_, kvc_, kpe_):
        active = jnp.minimum(cuq, cuq[dist[2]])
        nope_, rope_ = kcu.update_sparse_mla_kv_cache(
            nope_, rope_, kvc_, kpe_, kvl, pi, active, nope_spec=ns, rope_spec=rs)
        out = sparse_ragged_paged_attention(
            jnp.concatenate([ql_, qpe_], axis=-1), nope_, rope_, rows, None, cuq, dist,
            sm_scale=sm, k_scale=1.0, cache_layout="sparsecore", **KW)
        return out[..., :512], nope_, rope_

    sh = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    fmt = Format(Layout(major_to_minor=(0, 1, 2)), sh)
    jf = jax.jit(step, donate_argnums=(0, 1), in_shardings=(sh, sh, fmt, fmt, sh, sh))
    qa = jax.device_put(jnp.asarray(c["q_latent"]), fmt)
    qb = jax.device_put(jnp.asarray(c["q_rope"]), fmt)
    st = {"c": (nope, rope)}

    def call():
        out, no, ro = jf(*st["c"], qa, qb, kvc, kpe)
        st["c"] = (no, ro)
        return out

    for _ in range(3):
        call()
    jax.block_until_ready(st["c"][0])
    jax.profiler.start_trace(xp)
    for _ in range(3):
        call()
    jax.block_until_ready(st["c"][0])
    jax.profiler.stop_trace()

    files = glob.glob(os.path.join(xp, "**", "*.xplane.pb"), recursive=True)
    lines = []
    for path in files:
        for pname, names, plines in _planes(memoryview(open(path, "rb").read())):
            if pname.startswith("/device:TPU"):
                for lname, evs in plines:
                    lines.append((pname, lname,
                                  [(names.get(m, "?"), o, o + du) for m, o, du in evs]))
    first = min(p for p, l, _ in lines if l == "XLA Modules")
    mods = sorted(e for p, l, ev in lines if l == "XLA Modules" and p == first for e in ev
                  if e[2] - e[1] > 1e6)
    w0, w1 = mods[1][1], mods[1][2]
    gat, tc_ops, tgts = [], [], []
    for pname, lname, evs in lines:
        if lname in ("XLA Modules", "Steps"):
            continue
        for nm, s, e in evs:
            if e <= w0 or s >= w1:
                continue
            s, e = max(s, w0), min(e, w1)
            low = nm.lower()
            if any(k in nm for k in GATHER):
                gat.append((s, e))
            elif any(k in low for k in TARGETS) and e - s > 2e6:  # > 2 us
                tgts.append((nm, pname, lname, s, e))
            elif "SparseCore" not in pname and lname == "XLA Ops":
                tc_ops.append((s, e))
    us = lambda ps: round(ps / 1e6, 1)
    tot = dict(dur=0, sc=0, tc=0)
    seen = set()
    for nm, pname, lname, s, e in sorted(tgts, key=lambda x: x[3]):
        key = (nm, s)
        if key in seen:
            continue
        seen.add(key)
        sc, tc = inter([(s, e)], gat), inter([(s, e)], tc_ops)
        tot["dur"] += e - s
        tot["sc"] += sc
        tot["tc"] += tc
        print("TGT " + json.dumps({"n": nm[:70], "p": pname[-14:], "l": lname[:20],
                                   "t": us(s - w0), "d": us(e - s), "sc_busy_us": us(sc),
                                   "tc_other_us": us(tc)}), flush=True)
    busy = union(gat + tc_ops + [(s, e) for _, _, _, s, e in tgts])
    print("SUM " + json.dumps({"call_us": us(w1 - w0), "gather_union_us": us(union(gat)),
                               "targets_us": us(tot["dur"]),
                               "targets_under_gather_us": us(tot["sc"]),
                               "targets_overlap_tc_us": us(tot["tc"]),
                               "idle_us": us((w1 - w0) - busy)}), flush=True)
    print("TGT_DONE", flush=True)


if __name__ == "__main__":
    main()
