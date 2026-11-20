"""Test only: sglang-torchtpu main vs vllm-torchtpu HEAD sparse MLA (DSA) kernel.

The sglang kernel is ``sglang_tpu/attention/pallas/mla_sparse`` at this
branch's base (upstream/main, before #2246). The vllm kernel is
``vllm_torchtpu/kernels/mla`` at vllm-torchtpu ecb07423a, copied verbatim into
``vllm_mla/`` (only the package prefix in its imports is rewritten).

Each variant writes this step's KV into its cache with its own module's writer,
then runs its own ``sparse_ragged_paged_attention`` through the page table
(sequence-local top-k rows), the way each framework serves by default:

* ``sgl_main``: sglang, TensorCore cache layout, default chunk 64 / batch 16
  (what ``mla_sparse_fused.sparse_mla_prefill_fused`` passes on main).
* ``vllm_default``: vllm, SparseCore cache layout (``TPU_SPARSE_MLA_*_LAYOUT``
  default), default kernel kwargs (split_q, capped path, masked dense all off).
* ``vllm_tc``: vllm kernel, TensorCore layout: same layout as ``sgl_main``, so
  it isolates the code difference.
* ``sgl_sc``: sglang kernel, SparseCore layout: same layout as
  ``vllm_default``, again isolating the code difference.

Same seeded inputs for all. Reported per variant: device module time per call
(profiler, the primary number), median wall time, op categories, SparseCore TEC
busy time, bit-exactness vs ``sgl_main`` and vs the same-layout variant, and the
error vs f64 attention. Prints ``VVS`` lines. Production code is untouched.
"""

import argparse
import importlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_fp8_attention_ab as att  # noqa: E402

FP = att.FP = 64  # the PR 2198 baseline page size
CASES = ("pr2198_extend_c1", "pr2198_verify_b4", "pr2198_decode_b4")
CATEGORIES = [
    ("tc_attention", ("MLA-tensorcore", "MLA-sparsecore")),
    ("sc_gather", ("sc_dsa_gather", "dsa_gather", "native_sc", "sparse-core-data-format")),
    ("page_lookup", ("gather_page_ids",)),
    ("kv_store", ("scatter",)),
    ("q_prep", ("q_hbm_ref", "concatenate")),
    ("copy", ("copy",)),
]
MODULES = {
    "sgl": ("sglang_tpu.attention.pallas.mla_sparse.kv_cache_utils",
            "sglang_tpu.attention.pallas.mla_sparse.kernel"),
    "vllm": ("vllm_mla.mla.kv_cache_utils", "vllm_mla.mla.sparse.kernel"),
}
# (name, module, layout, same-layout partner for the code-only exactness check)
VARIANTS = (
    ("sgl_main", "sgl", "tensorcore", "vllm_tc"),
    ("vllm_default", "vllm", "sparsecore", "sgl_sc"),
    ("vllm_tc", "vllm", "tensorcore", "sgl_main"),
    ("sgl_sc", "sgl", "sparsecore", "vllm_default"),
)


def emit(**kw):
    print("VVS " + json.dumps(kw), flush=True)


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

    from xplane_summary import breakdown, summarize

    mods = {k: (importlib.import_module(u), importlib.import_module(kn))
            for k, (u, kn) in MODULES.items()}
    emit(info="modules", files={k: [m.__file__ for m in v] for k, v in mods.items()})

    f8 = jnp.float8_e4m3fn
    fmax = float(jnp.finfo(f8).max)
    q8 = lambda x: jnp.clip(x.astype(jnp.float32), -fmax, fmax).astype(f8)

    def specs(kcu, layout, P):
        lay = {"tensorcore": kcu.KVCacheLayout.TENSORCORE,
               "sparsecore": kcu.KVCacheLayout.SPARSECORE}[layout]
        return (kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.NOPE, lay, P, FP, 512),
                kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.ROPE, lay, P, FP, 64))

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
        outs = {}
        emit(case=name, T=T, pages=P)

        for vname, mk, layout, partner in VARIANTS:
            kcu, kmod = mods[mk]
            row = {"case": name, "variant": vname, "module": mk, "layout": layout}
            try:
                ns, rs = specs(kcu, layout, P)
                fill = jax.jit(lambda nope, rope, kv_, pe_, _k=kcu, _ns=ns, _rs=rs:
                               _k.update_sparse_mla_kv_cache_jax(
                                   nope, rope, kv_, pe_, jnp.array([n], jnp.int32),
                                   jnp.arange(P, dtype=jnp.int32), jnp.array([0, n], jnp.int32),
                                   nope_spec=_ns, rope_spec=_rs))
                cache0 = fill(jnp.zeros(ns.shape, ns.jax_dtype),
                              jnp.zeros(rs.shape, rs.jax_dtype), kv0, pe0)
                cache0 = tuple(np.asarray(x) for x in cache0)

                def step(nope, rope, q_, kvc_, kpe_, _k=kcu, _m=kmod, _ns=ns, _rs=rs, _l=layout):
                    active = jnp.minimum(cuq, cuq[dist[2]])
                    nope, rope = _k.update_sparse_mla_kv_cache(
                        nope, rope, kvc_, kpe_, kvl, pi, active, nope_spec=_ns, rope_spec=_rs)
                    out = _m.sparse_ragged_paged_attention(
                        q_, nope, rope, topk, pi, cuq, dist,
                        sm_scale=sm, k_scale=1.0, cache_layout=_l)
                    return out[..., :512], nope, rope

                step.__name__ = f"vvs_{vname}"
                jf = jax.jit(step, donate_argnums=(0, 1))
                st = {"c": tuple(jnp.asarray(x) for x in cache0)}

                def call():
                    out, no, ro = jf(*st["c"], q, kvc, kpe)
                    st["c"] = (no, ro)
                    return out

                out = np.asarray(call())
                outs[vname] = out
                for ref in ("sgl_main", partner):
                    if ref in outs and ref != vname:
                        row[f"exact_vs_{ref}"] = bool(np.array_equal(
                            out.view(np.uint16), outs[ref].view(np.uint16)))
                row["error"] = att.error_vs_bf16(c, out.astype(np.float64), rows_bf)
                sync = lambda: jax.block_until_ready(st["c"][0])
                med, runs = att._median_time(call, sync, a.iters, a.repeats)
                row["wall_us"] = round(med, 1)
                row["wall_us_range"] = [round(min(runs), 1), round(max(runs), 1)]
                d = f"{a.xprof_dir}/vvs_{name}_{vname}"
                sync()
                jax.profiler.start_trace(d)
                for _ in range(a.iters):
                    call()
                sync()
                jax.profiler.stop_trace()
                x = summarize(d, a.iters, top=8)
                row["module_us"] = round(sum(u for _, u in x.get("modules", [])), 1)
                bd = breakdown(d, a.iters, CATEGORIES)
                tec = [v for k, v in bd.get("lines", {}).items() if "| TEC" in k]
                row["categories"] = bd.get("categories")
                row["sc_tec_busy_us_max"] = round(max(tec), 1) if tec else None
                row["top_ops"] = [(nm[:60], round(u, 1)) for nm, u in x.get("ops", [])[:6]]
            except Exception as e:  # report, keep the other variants
                import traceback

                traceback.print_exc()
                row["fail"] = repr(e)[:800]
            emit(**row)
    print("VVS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
