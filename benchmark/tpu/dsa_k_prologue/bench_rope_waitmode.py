"""Test only: does moving the RoPE transpose off SparseCore shorten the gather?

The native SC gather (``native_sc_gather.py``) DMAs the gathered NOPE rows and,
for RoPE, runs ``transpose_quad`` (register delta-swaps) into the four-token TC
format. ``wait_mode`` controls the overlap:

* ``overlap_rope`` (production): start the NOPE DMA, run the RoPE transpose, then
  wait the NOPE DMA -- the transpose hides under the 8x-larger NOPE DMA.
* ``before_rope``: wait the NOPE DMA first, then transpose -- serial.

``T_before - T_overlap`` of the per-chunk SC gather is how much of the RoPE
transpose is *not* already hidden, i.e. the ceiling on what moving it to the
TC VPU could save on the gather. If it is ~0 the gather is NOPE-DMA bound and
the move buys nothing. Reads the SC/TC timeline (one SC tile traced, so the TC
attention chunks survive) and reports per-chunk gather / attention busy time.
Prints ``WM`` lines. nope_unpack stays ``reshape`` (production). Output of the
two modes must match byte-for-byte.
"""

import argparse
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_vperm_unpack as bvu  # noqa: E402 (sets LIBTPU_INIT_ARGS, imports att)

import numpy as np  # noqa: E402

att = bvu.att
FP = att.FP = 64
CASES = ("pr2198_extend_c1", "pr2198_decode_b4")
MODES = ("overlap_rope", "before_rope")


def emit(**kw):
    print("WM " + json.dumps(kw), flush=True)


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def chunks(iv):
    """Per-interval durations after merging touching pieces."""
    return [round(e - s, 2) for s, e in merge(iv)]


def analyze(trace_dir, call):
    from xplane_timeline import call_events

    r = call_events(trace_dir, call=call)
    if "error" in r:
        return r
    tc, sc = r["tc"], r["sc"]
    attn = chunks([[s, e] for n, s, e in tc if "MLA-" in n])
    gat = chunks([[s, e] for n, s, e in sc
                  if "native_sc_gather" in n and not n.startswith("TEC")])
    tec = chunks([[s, e] for n, s, e in sc if n.startswith("TEC")])
    stead = lambda xs: round(st.median(xs[1:]), 2) if len(xs) > 1 else (xs[0] if xs else None)
    return {"module_us": r["module_us"], "n_gather": len(gat), "n_attn": len(attn),
            "gather_sum": round(sum(gat), 1), "gather_per_chunk": stead(gat),
            "gather_first": gat[0] if gat else None,
            "attn_sum": round(sum(attn), 1), "attn_per_chunk": stead(attn),
            "tec_busy_sum": round(sum(tec), 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=list(CASES))
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--xprof-dir", required=True)
    p.add_argument("--traces", type=int, default=2)
    a = p.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    from vllm_mla.mla import kv_cache_utils as kcu
    from vllm_mla.mla.sparse import kernel as kmod

    emit(info="modules", kernel=kmod.__file__, jax=jax.__version__)
    f8 = jnp.float8_e4m3fn
    fmax = float(jnp.finfo(f8).max)
    q8 = lambda x: jnp.clip(x.astype(jnp.float32), -fmax, fmax).astype(f8)
    lay = kcu.KVCacheLayout.SPARSECORE

    for name in a.cases:
        c = att.case(name)
        P, T, S = c["P"], c["T"], c["S"]
        n = P * FP
        q = jnp.concatenate([jnp.asarray(c["q_latent"]), jnp.asarray(c["q_rope"])], axis=-1)
        kvc, kpe = q8(jnp.asarray(c["kv_c"])), q8(jnp.asarray(c["k_pe"]))
        kv0, pe0 = q8(jnp.asarray(c["prefix_kv"])), q8(jnp.asarray(c["prefix_pe"]))
        kvl, pi, cuq = (jnp.asarray(c[k]) for k in ("kv", "pi", "cu_q"))
        dist = jnp.asarray(np.array([S] * 3 if c["mode"] == "decode" else [0, 0, S], np.int32))
        topk = jnp.asarray(c["topk"])
        sm = float(c["sm_scale"])
        ns = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.NOPE, lay, P, FP, 512)
        rs = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.ROPE, lay, P, FP, 64)
        fill = jax.jit(lambda nope, rope, kv_, pe_: kcu.update_sparse_mla_kv_cache_jax(
            nope, rope, kv_, pe_, jnp.array([n], jnp.int32), jnp.arange(P, dtype=jnp.int32),
            jnp.array([0, n], jnp.int32), nope_spec=ns, rope_spec=rs))
        cache0 = tuple(np.asarray(x) for x in fill(
            jnp.zeros(ns.shape, ns.jax_dtype), jnp.zeros(rs.shape, rs.jax_dtype), kv0, pe0))
        emit(case=name, T=T, pages=P)
        outs = {}

        for mode in MODES:
            row = {"case": name, "mode": mode}
            try:
                def step(nope, rope, q_, kvc_, kpe_, _m=mode):
                    active = jnp.minimum(cuq, cuq[dist[2]])
                    nope, rope = kcu.update_sparse_mla_kv_cache(
                        nope, rope, kvc_, kpe_, kvl, pi, active, nope_spec=ns, rope_spec=rs)
                    out = kmod.sparse_ragged_paged_attention(
                        q_, nope, rope, topk, pi, cuq, dist, sm_scale=sm, k_scale=1.0,
                        cache_layout="sparsecore", nope_unpack="reshape",
                        sparsecore_gather_wait_mode=_m)
                    return out[..., :512], nope, rope

                step.__name__ = f"wm_{mode}"
                jf = jax.jit(step, donate_argnums=(0, 1))
                stt = {"c": tuple(jnp.asarray(x) for x in cache0)}

                def call():
                    out, no, ro = jf(*stt["c"], q, kvc, kpe)
                    stt["c"] = (no, ro)
                    return out

                out = np.asarray(call())
                outs[mode] = out
                if "overlap_rope" in outs and mode != "overlap_rope":
                    row["exact_vs_overlap"] = bool(np.array_equal(
                        out.view(np.uint16), outs["overlap_rope"].view(np.uint16)))
                sync = lambda: jax.block_until_ready(stt["c"][0])
                med, runs = att._median_time(call, sync, a.iters, a.repeats)
                row["wall_us"] = round(med, 1)
                tl = []
                for t in range(a.traces):
                    d = f"{a.xprof_dir}/wm_{name}_{mode}_{t}"
                    sync()
                    opts = jax.profiler.ProfileOptions()
                    opts.advanced_configuration = {"tpu_num_sparse_core_tiles_to_trace": 1}
                    jax.profiler.start_trace(d, profiler_options=opts)
                    for _ in range(2):
                        call()
                    sync()
                    jax.profiler.stop_trace()
                    tl.append(analyze(d, call=1))
                row["timeline"] = tl
            except Exception as e:  # report, keep the other mode
                import traceback

                traceback.print_exc()
                row["fail"] = repr(e)[:800]
            emit(**row)
    print("WM_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
