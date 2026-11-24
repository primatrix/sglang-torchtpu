"""Test only: VPU-only / MXU-only / DMA-only ablation of the vllm sparse MLA kernel.

Following the CSA2 roofline write-up, the vllm-torchtpu kernel copied in
``vllm_mla/`` gets a trace-time switch (``kernel._ABLATE``) that keeps only one
unit's work in the uncapped attention body:

* ``full``: unchanged kernel.
* ``dma``: the q / KV fetches, output sends and their waits; q is copied to
  the output, no unpack, no matmul, no softmax.
* ``mxu``: QK and PV matmuls on a VMEM operand already in compute layout
  (fp8[bkv_sz, 640]); no unpack, no softmax. PV still takes the f32 scores.
* ``vpu``: unpack (kept live by a store), mask, softmax, normalize and the
  output cast; scores are a broadcast q column, PV is a tile of p. No matmul.
* ``vpu_repack``: ``vpu`` plus a forced s8->s32->s8 round trip of the KV
  bytes. A plain store lets the compiler skip the byte relayout that the MXU
  push needs (``vpu`` ran 16 ``vunpack`` per query vs 1,280 in ``full``); this
  stands in for it with a similar instruction count, not the same shuffles.

Each mode runs the same step as ``bench_vllm_vs_sglang.py`` (KV write, then
``sparse_ragged_paged_attention``) on the vllm-default layout for that shape,
and reports the module time, the ``MLA-`` attention kernel's own time
(profiler category ``tc_attention``), and the LLO counts of that kernel
(``LLOSTAT`` from ``bench_vllm_attn_llo``). Outputs of the ablated modes are
not meaningful. Prints ``ABL`` lines. Production code is untouched.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench_vllm_attn_llo as llo  # noqa: E402  (sets the LLO dump flags before jax)
import numpy as np  # noqa: E402

import bench_fp8_attention_ab as att  # noqa: E402

FP = att.FP = 64
MODES = ("full", "dma", "mxu", "vpu", "vpu_repack")
# vllm's default layout per shape; verify_b4 fails on the SparseCore layout.
CASES = {"pr2198_extend_c1": "sparsecore", "pr2198_verify_b4": "tensorcore"}
CATEGORIES = [
    ("tc_attention", ("MLA-tensorcore", "MLA-sparsecore")),
    ("sc_gather", ("sc_dsa_gather", "dsa_gather", "native_sc", "sparse-core-data-format")),
    ("page_lookup", ("gather_page_ids",)),
    ("kv_store", ("scatter",)),
]


def emit(**kw):
    print("ABL " + json.dumps(kw), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=list(CASES))
    p.add_argument("--modes", nargs="+", default=list(MODES))
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--xprof-dir", required=True)
    a = p.parse_args()

    import jax
    import jax.numpy as jnp

    from vllm_mla.mla import kv_cache_utils as kcu
    from vllm_mla.mla.sparse import kernel as kmod
    from xplane_summary import breakdown, summarize

    f8 = jnp.float8_e4m3fn
    fmax = float(jnp.finfo(f8).max)
    q8 = lambda x: jnp.clip(x.astype(jnp.float32), -fmax, fmax).astype(f8)

    for name in a.cases:
        layout = CASES[name]
        c = att.case(name)
        P, T, S = c["P"], c["T"], c["S"]
        n = P * FP
        kv0, pe0 = q8(jnp.asarray(c["prefix_kv"])), q8(jnp.asarray(c["prefix_pe"]))
        q = jnp.concatenate([jnp.asarray(c["q_latent"]), jnp.asarray(c["q_rope"])], axis=-1)
        kvc, kpe = q8(jnp.asarray(c["kv_c"])), q8(jnp.asarray(c["k_pe"]))
        kvl, pi, cuq = (jnp.asarray(c[k]) for k in ("kv", "pi", "cu_q"))
        dist = jnp.asarray(np.array([S] * 3 if c["mode"] == "decode" else [0, 0, S], np.int32))
        topk = jnp.asarray(c["topk"])
        sm = float(c["sm_scale"])
        lay = {"tensorcore": kcu.KVCacheLayout.TENSORCORE,
               "sparsecore": kcu.KVCacheLayout.SPARSECORE}[layout]
        ns = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.NOPE, lay, P, FP, 512)
        rs = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.ROPE, lay, P, FP, 64)
        kmod._ABLATE = None
        fill = jax.jit(lambda nope, rope, kv_, pe_: kcu.update_sparse_mla_kv_cache_jax(
            nope, rope, kv_, pe_, jnp.array([n], jnp.int32), jnp.arange(P, dtype=jnp.int32),
            jnp.array([0, n], jnp.int32), nope_spec=ns, rope_spec=rs))
        cache0 = tuple(np.asarray(x) for x in fill(
            jnp.zeros(ns.shape, ns.jax_dtype), jnp.zeros(rs.shape, rs.jax_dtype), kv0, pe0))
        emit(case=name, layout=layout, T=T, pages=P)

        for mode in a.modes:
            tag = f"{name}_{mode}"
            row = {"case": name, "mode": mode, "layout": layout, "T": T}
            try:
                kmod._ABLATE = None if mode == "full" else mode
                jax.clear_caches()

                def step(nope, rope, q_, kvc_, kpe_):
                    active = jnp.minimum(cuq, cuq[dist[2]])
                    nope, rope = kcu.update_sparse_mla_kv_cache(
                        nope, rope, kvc_, kpe_, kvl, pi, active, nope_spec=ns, rope_spec=rs)
                    out = kmod.sparse_ragged_paged_attention(
                        q_, nope, rope, topk, pi, cuq, dist,
                        sm_scale=sm, k_scale=1.0, cache_layout=layout)
                    return out[..., :512], nope, rope

                step.__name__ = f"abl_{mode}"
                jf = jax.jit(step, donate_argnums=(0, 1))
                st = {"c": tuple(jnp.asarray(x) for x in cache0)}

                def call():
                    out, no, ro = jf(*st["c"], q, kvc, kpe)
                    st["c"] = (no, ro)
                    return out

                before = llo.files_now()
                out = np.asarray(call())
                llo.report(tag, llo.files_now() - before)
                row["finite_frac"] = float(np.isfinite(out.astype(np.float32)).mean())
                sync = lambda: jax.block_until_ready(st["c"][0])
                med, runs = att._median_time(call, sync, a.iters, a.repeats)
                row["wall_us"] = round(med, 1)
                d = f"{a.xprof_dir}/abl_{name}_{mode}"
                sync()
                jax.profiler.start_trace(d)
                for _ in range(a.iters):
                    call()
                sync()
                jax.profiler.stop_trace()
                x = summarize(d, a.iters, top=8)
                row["module_us"] = round(sum(u for _, u in x.get("modules", [])), 1)
                bd = breakdown(d, a.iters, CATEGORIES)
                row["categories"] = bd.get("categories")
                row["module_span_us"] = bd.get("module_span_us")
                row["ops_busy_us"] = bd.get("ops_busy_us")
                attn = (bd.get("categories") or {}).get("tc_attention")
                row["attn_us_per_query"] = (
                    round(attn / T, 4) if isinstance(attn, (int, float)) else None)
                row["top_ops"] = [(nm[:60], round(u, 1)) for nm, u in x.get("ops", [])[:6]]
            except Exception as e:  # report, keep the other modes
                import traceback

                traceback.print_exc()
                row["fail"] = repr(e)[:800]
            emit(**row)
    kmod._ABLATE = None
    print("ABL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
