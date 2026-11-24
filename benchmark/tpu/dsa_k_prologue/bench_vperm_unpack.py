"""Test only: vllm-torchtpu sparse MLA kernel, NOPE unpack reshape vs vperm.slane.

Runs the vllm kernel copy in ``vllm_mla/`` (SparseCore cache layout, the vllm
default) twice per case: ``nope_unpack="reshape"`` (production,
``bitcast(u8).reshape``) and ``nope_unpack="vperm"`` (one vperm.slane per
gathered 8-token vreg in VMEM, then stride-4 sublane loads; see
``_vperm_unpack_nope``). Same seeded inputs, same step as
``bench_vllm_vs_sglang.py`` (cache write + attention through the page table).

Reported per variant: device module time per call (profiler), median wall time,
op categories, bit-exactness vs the reshape variant, error vs f64 attention,
and the LLO opcode histogram of the attention kernel's final bundles. Prints
``VPU`` / ``LLOSTAT`` lines. Production code is untouched.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# Sets LIBTPU_INIT_ARGS for the LLO text dump; must precede the jax import.
import bench_vllm_attn_llo as llo  # noqa: E402

import numpy as np  # noqa: E402

import bench_fp8_attention_ab as att  # noqa: E402

FP = att.FP = 64
CASES = ("pr2198_extend_c1", "pr2198_verify_b4", "pr2198_decode_b4")
CATEGORIES = [
    ("tc_attention", ("MLA-tensorcore", "MLA-sparsecore")),
    ("sc_gather", ("sc_dsa_gather", "dsa_gather", "native_sc", "sparse-core-data-format")),
    ("page_lookup", ("gather_page_ids",)),
    ("kv_store", ("scatter",)),
    ("q_prep", ("q_hbm_ref", "concatenate")),
    ("copy", ("copy",)),
]
VARIANTS = ("reshape", "vperm")


def emit(**kw):
    print("VPU " + json.dumps(kw), flush=True)


def _emit_chunked(kind, head, items, limit=3200):
    """Print ``items`` as several ``kind`` lines, each JSON under ``limit`` chars."""
    part, size, idx = [], 0, 0
    for it in items:
        n = len(json.dumps(it)) + 1
        if part and size + n > limit:
            print(f"{kind} " + json.dumps({**head, "part": idx, "ev": part}), flush=True)
            part, size, idx = [], 0, idx + 1
        part.append(it)
        size += n
    print(f"{kind} " + json.dumps({**head, "part": idx, "ev": part, "last": True}), flush=True)


def dump_timeline(jax, call, sync, base, case, variant, traces):
    """Each trace holds two calls; the second (warm) is dumped.

    With all SparseCore tiles traced the TEC lines carry ~800k events and the
    TC "XLA Ops" line loses most attention chunks (5 of 48 kept in
    exp-qcw0lbk9bs's layer-0 trace), so a second config traces one SC tile.
    """
    from xplane_timeline import call_events

    configs = (("default", None), ("sc_tile1", {"tpu_num_sparse_core_tiles_to_trace": 1}))
    for cfg, adv in configs:
        for t in range(traces):
            d = f"{base}_{cfg}_{t}"
            head = {"case": case, "variant": variant, "cfg": cfg, "trace": t}
            opts = None
            if adv is not None:
                opts = jax.profiler.ProfileOptions()
                opts.advanced_configuration = adv
            sync()
            try:
                jax.profiler.start_trace(d, profiler_options=opts)
            except Exception as e:  # unknown option: report, keep the other config
                print("TLM " + json.dumps({**head, "fail": repr(e)[:400]}), flush=True)
                break
            for _ in range(2):
                call()
            sync()
            jax.profiler.stop_trace()
            r = call_events(d, call=1)
            if "error" in r:
                print("TLM " + json.dumps({**head, **r}), flush=True)
                continue
            n_mla = sum("MLA-" in n for n, _, _ in r["tc"])
            print("TLM " + json.dumps({**head, "module_us": r["module_us"],
                                        "n_modules": r["n_modules"], "n_tc": len(r["tc"]),
                                        "n_mla": n_mla, "n_sc": len(r["sc"]),
                                        "dropped_small_us": r["dropped_small_us"]}), flush=True)
            _emit_chunked("TLE", {**head, "plane": "tc"}, r["tc"])
            _emit_chunked("TLE", {**head, "plane": "sc"}, r["sc"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=list(CASES))
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--xprof-dir", required=True)
    p.add_argument("--timeline", action="store_true",
                   help="also dump per-op absolute TC/SC intervals of single warm calls (TLE lines)")
    p.add_argument("--timeline-traces", type=int, default=2)
    a = p.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    from vllm_mla.mla import kv_cache_utils as kcu
    from vllm_mla.mla.sparse import kernel as kmod
    from xplane_summary import breakdown, summarize

    emit(info="modules", kernel=kmod.__file__, jax=jax.__version__)
    f8 = jnp.float8_e4m3fn
    fmax = float(jnp.finfo(f8).max)
    q8 = lambda x: jnp.clip(x.astype(jnp.float32), -fmax, fmax).astype(f8)
    lay = kcu.KVCacheLayout.SPARSECORE

    for name in a.cases:
        c = att.case(name)
        P, T, S = c["P"], c["T"], c["S"]
        n = P * FP
        rows_bf = att.bf16_rows(c)
        kv0, pe0 = q8(jnp.asarray(c["prefix_kv"])), q8(jnp.asarray(c["prefix_pe"]))
        q = jnp.concatenate([jnp.asarray(c["q_latent"]), jnp.asarray(c["q_rope"])], axis=-1)
        kvc, kpe = q8(jnp.asarray(c["kv_c"])), q8(jnp.asarray(c["k_pe"]))
        kvl, pi, cuq = (jnp.asarray(c[k]) for k in ("kv", "pi", "cu_q"))
        dist = jnp.asarray(np.array([S] * 3 if c["mode"] == "decode" else [0, 0, S], np.int32))
        topk = jnp.asarray(c["topk"])
        sm = float(c["sm_scale"])
        ns = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.NOPE, lay, P, FP, 512)
        rs = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.ROPE, lay, P, FP, 64)
        fill = jax.jit(lambda nope, rope, kv_, pe_: kcu.update_sparse_mla_kv_cache_jax(
            nope, rope, kv_, pe_, jnp.array([n], jnp.int32), jnp.arange(P, dtype=jnp.int32),
            jnp.array([0, n], jnp.int32), nope_spec=ns, rope_spec=rs))
        cache0 = fill(jnp.zeros(ns.shape, ns.jax_dtype), jnp.zeros(rs.shape, rs.jax_dtype), kv0, pe0)
        cache0 = tuple(np.asarray(x) for x in cache0)
        emit(case=name, T=T, pages=P)
        outs = {}

        for variant in VARIANTS:
            row = {"case": name, "variant": variant}
            try:
                def step(nope, rope, q_, kvc_, kpe_, _v=variant):
                    active = jnp.minimum(cuq, cuq[dist[2]])
                    nope, rope = kcu.update_sparse_mla_kv_cache(
                        nope, rope, kvc_, kpe_, kvl, pi, active, nope_spec=ns, rope_spec=rs)
                    out = kmod.sparse_ragged_paged_attention(
                        q_, nope, rope, topk, pi, cuq, dist, sm_scale=sm, k_scale=1.0,
                        cache_layout="sparsecore", nope_unpack=_v)
                    return out[..., :512], nope, rope

                step.__name__ = f"vpu_{variant}"
                jf = jax.jit(step, donate_argnums=(0, 1))
                st = {"c": tuple(jnp.asarray(x) for x in cache0)}

                def call():
                    out, no, ro = jf(*st["c"], q, kvc, kpe)
                    st["c"] = (no, ro)
                    return out

                before = llo.files_now()
                out = np.asarray(call())
                llo.report(f"{name}_{variant}", llo.files_now() - before)
                outs[variant] = out
                if variant != "reshape" and "reshape" in outs:
                    row["exact_vs_reshape"] = bool(np.array_equal(
                        out.view(np.uint16), outs["reshape"].view(np.uint16)))
                row["error"] = att.error_vs_bf16(c, out.astype(np.float64), rows_bf)
                sync = lambda: jax.block_until_ready(st["c"][0])
                med, runs = att._median_time(call, sync, a.iters, a.repeats)
                row["wall_us"] = round(med, 1)
                row["wall_us_range"] = [round(min(runs), 1), round(max(runs), 1)]
                d = f"{a.xprof_dir}/vpu_{name}_{variant}"
                sync()
                jax.profiler.start_trace(d)
                for _ in range(a.iters):
                    call()
                sync()
                jax.profiler.stop_trace()
                x = summarize(d, a.iters, top=8)
                row["module_us"] = round(sum(u for _, u in x.get("modules", [])), 1)
                row["categories"] = breakdown(d, a.iters, CATEGORIES).get("categories")
                row["top_ops"] = [(nm[:60], round(u, 1)) for nm, u in x.get("ops", [])[:6]]
                if a.timeline:
                    dump_timeline(jax, call, sync, f"{a.xprof_dir}/tl_{name}_{variant}",
                                  name, variant, a.timeline_traces)
            except Exception as e:  # report, keep the other variants
                import traceback

                traceback.print_exc()
                row["fail"] = repr(e)[:1500]
            emit(**row)
    print("VPU_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
