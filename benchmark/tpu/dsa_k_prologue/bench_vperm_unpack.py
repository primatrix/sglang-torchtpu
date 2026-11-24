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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=list(CASES))
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--xprof-dir", required=True)
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
            except Exception as e:  # report, keep the other variants
                import traceback

                traceback.print_exc()
                row["fail"] = repr(e)[:1500]
            emit(**row)
    print("VPU_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
