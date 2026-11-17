"""Target sparse MLA (S5-S6) parameter sweep at the PR 2198 shapes, page 64.

JAX on TPU (the torch op runs the same JAX function). Every variant writes this
step's KV into its cache, then runs ``sparse_ragged_paged_attention`` on the same
picked rows (seeded, 2048 per query):

* ``target``: the PR 2198 baseline as served: TensorCore cache layout, sequence-local
  rows through the page table, default kernel parameters.
* ``tc_phys_*``: TensorCore layout, flat cache slots (no page-table lookup), sweeping
  ``gather_and_attention_chunk_size`` (``c``) and ``attention_kernel_batch_size``
  (``b``). In this layout the gather is ``dsa_gather.dsa_gather``; the
  ``sparsecore_*`` parameters are not read.
* ``sc_phys_*``: SparseCore cache layout (uint32 rows, native SparseCore gather
  ``dsa_gather_native_sc``), flat slots, sweeping the chunk size, ``atoms``
  (``sparsecore_atoms_per_batch``) and ``wait`` (``sparsecore_gather_wait_mode``).

Reported per variant: median wall time and range, device module time, the op
breakdown with every TPU line's busy time (SparseCore TECs included), the error vs
f64 attention on the unquantized K/V, and whether the output equals ``target``'s
bit for bit. A variant that does not compile is reported and skipped.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_fp8_attention_ab as att  # noqa: E402
from check_physical_topk import to_physical  # noqa: E402

FP = att.FP = 64  # the PR 2198 baseline page size
CASES = ("pr2198_extend_c1", "pr2198_verify_b4", "pr2198_decode_b4")
CATEGORIES = [
    ("tc_attention", ("MLA-tensorcore", "MLA-sparsecore")),
    ("sc_gather", ("sc_dsa_gather", "dsa_gather", "native_sc", "sparse-core-data-format")),
    ("page_lookup", ("gather_page_ids",)),
    ("kv_store", ("scatter",)),
    ("q_prep", ("q_hbm_ref", "concatenate")),
    ("out_slice", ("slice",)),
    ("copy", ("copy",)),
]


def variants(T):
    """``(name, layout, physical, kwargs)``; ``target`` first."""
    out = [("target", "tensorcore", False, {}), ("tc_phys_default", "tensorcore", True, {})]
    if T >= 128:
        for c in (64, 128, 256, 512):
            for b in (8, 16, 32):
                if (c, b) != (64, 16):
                    out.append((f"tc_phys_c{c}_b{b}", "tensorcore", True,
                                dict(gather_and_attention_chunk_size=c,
                                     attention_kernel_batch_size=b)))
        for c in (64, 128, 256):
            for atoms in (4, 8, 16):
                for wait in ("overlap_rope", "before_rope"):
                    out.append((f"sc_phys_c{c}_a{atoms}_{wait}", "sparsecore", True,
                                dict(gather_and_attention_chunk_size=c,
                                     sparsecore_atoms_per_batch=atoms,
                                     sparsecore_gather_wait_mode=wait)))
        # Chunk x attention kernel batch at the fastest gather setting (a16,
        # overlap_rope): isolates why chunk 256 was slower than 128 (exp-cdkg2zm04b).
        for c in (128, 256, 512):
            for b in (8, 16, 32):
                out.append((f"sc_phys_c{c}_b{b}", "sparsecore", True,
                            dict(gather_and_attention_chunk_size=c,
                                 attention_kernel_batch_size=b,
                                 sparsecore_atoms_per_batch=16,
                                 sparsecore_gather_wait_mode="overlap_rope")))
    else:
        for b in (1, 2, 4, 8):
            out.append((f"tc_phys_b{b}", "tensorcore", True,
                        dict(attention_kernel_batch_size=b, small_chunk_batch_size=b)))
        for atoms in (1, 2, 4, 8, 16):
            for wait in ("overlap_rope", "before_rope"):
                out.append((f"sc_phys_a{atoms}_{wait}", "sparsecore", True,
                            dict(sparsecore_atoms_per_batch=atoms,
                                 small_chunk_gather_atoms=atoms,
                                 sparsecore_gather_wait_mode=wait)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=list(CASES))
    p.add_argument("--only", nargs="*", default=None, help="variant names to run")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--xprof-dir", required=True)
    a = p.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    from xplane_summary import breakdown, summarize
    from sglang_tpu.attention.pallas.mla_sparse import kv_cache_utils as kcu
    from sglang_tpu.attention.pallas.mla_sparse.kernel import sparse_ragged_paged_attention

    f8 = jnp.float8_e4m3fn
    fmax = float(jnp.finfo(f8).max)
    q8 = lambda x: jnp.clip(x.astype(jnp.float32), -fmax, fmax).astype(f8)
    layouts = {"tensorcore": kcu.KVCacheLayout.TENSORCORE,
               "sparsecore": kcu.KVCacheLayout.SPARSECORE}

    def specs(layout, P):
        return (kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.NOPE, layouts[layout], P, FP, 512),
                kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.ROPE, layouts[layout], P, FP, 64))

    for name in a.cases:
        c = att.case(name)
        P, T, S = c["P"], c["T"], c["S"]
        n = P * FP
        rows_bf = att.bf16_rows(c)
        kv0, pe0 = q8(jnp.asarray(c["prefix_kv"])), q8(jnp.asarray(c["prefix_pe"]))
        caches0 = {}
        for layout in layouts:
            ns, rs = specs(layout, P)
            fill = jax.jit(lambda nope, rope, kv_, pe_, _ns=ns, _rs=rs:
                           kcu.update_sparse_mla_kv_cache_jax(
                               nope, rope, kv_, pe_, jnp.array([n], jnp.int32),
                               jnp.arange(P, dtype=jnp.int32), jnp.array([0, n], jnp.int32),
                               nope_spec=_ns, rope_spec=_rs))
            nope, rope = fill(jnp.zeros(ns.shape, ns.jax_dtype), jnp.zeros(rs.shape, rs.jax_dtype),
                              kv0, pe0)
            caches0[layout] = (np.asarray(nope), np.asarray(rope))
        q = jnp.concatenate([jnp.asarray(c["q_latent"]), jnp.asarray(c["q_rope"])], axis=-1)
        kvc, kpe = q8(jnp.asarray(c["kv_c"])), q8(jnp.asarray(c["k_pe"]))
        kvl, pi, cuq = (jnp.asarray(c[k]) for k in ("kv", "pi", "cu_q"))
        dist = jnp.asarray(np.array([S] * 3 if c["mode"] == "decode" else [0, 0, S], np.int32))
        rows = {False: jnp.asarray(c["topk"]),
                True: jnp.asarray(to_physical(c["topk"], c["pi"], c["ppr"], c["cu_q"], page=FP))}
        sm = float(c["sm_scale"])
        ref_out = None
        print("SWEEPCASE " + json.dumps({"case": name, "T": T, "pages": P}), flush=True)

        for vname, layout, physical, kw in variants(T):
            if a.only and vname not in a.only and vname != "target":
                continue
            row = {"case": name, "variant": vname, "layout": layout, "physical": physical, **kw}
            try:
                ns, rs = specs(layout, P)

                def step(nope, rope, q_, kvc_, kpe_, _ns=ns, _rs=rs, _l=layout, _p=physical,
                         _kw=kw):
                    active = jnp.minimum(cuq, cuq[dist[2]])
                    nope, rope = kcu.update_sparse_mla_kv_cache(
                        nope, rope, kvc_, kpe_, kvl, pi, active, nope_spec=_ns, rope_spec=_rs)
                    out = sparse_ragged_paged_attention(
                        q_, nope, rope, rows[_p], None if _p else pi, cuq, dist,
                        sm_scale=sm, k_scale=1.0, cache_layout=_l, **_kw)
                    return out[..., :512], nope, rope

                jf = jax.jit(step, donate_argnums=(0, 1))
                st = {"c": tuple(jnp.asarray(x) for x in caches0[layout])}

                def call():
                    out, no, ro = jf(*st["c"], q, kvc, kpe)
                    st["c"] = (no, ro)
                    return out

                out = np.asarray(call())
                if vname == "target":
                    ref_out = out
                row["exact_vs_target"] = (bool(np.array_equal(out.view(np.uint16),
                                                              ref_out.view(np.uint16)))
                                          if ref_out is not None else None)
                row["error"] = att.error_vs_bf16(c, out.astype(np.float64), rows_bf)
                sync = lambda: jax.block_until_ready(st["c"][0])
                med, runs = att._median_time(call, sync, a.iters, a.repeats)
                row["us"] = round(med, 1)
                row["us_range"] = [round(min(runs), 1), round(max(runs), 1)]
                d = f"{a.xprof_dir}/sweep_{name}_{vname}"
                sync()
                jax.profiler.start_trace(d)
                for _ in range(a.iters):
                    call()
                sync()
                jax.profiler.stop_trace()
                x = summarize(d, a.iters, top=8)
                row["module_us"] = round(sum(u for _, u in x.get("modules", [])), 1)
                bd = breakdown(d, a.iters, CATEGORIES)
                lines = bd.get("lines", {})
                tec = [v for k, v in lines.items() if "| TEC" in k]
                row["categories"] = bd.get("categories")
                row["tc_ops_busy_us"] = bd.get("ops_busy_us")
                row["sc_tec_busy_us_max"] = round(max(tec), 1) if tec else None
                row["sc_tec_busy_us_mean"] = round(sum(tec) / len(tec), 1) if tec else None
                row["top_ops"] = [(nm[:60], round(u, 1)) for nm, u in x.get("ops", [])[:6]]
            except Exception as e:  # a tile that does not compile or fit: report, go on
                row["fail"] = repr(e)[:400]
            print("SWEEP " + json.dumps(row), flush=True)
    print("SWEEP_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
