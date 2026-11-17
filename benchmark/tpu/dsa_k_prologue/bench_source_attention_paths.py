"""Source DSA attention (S5-S6) by precision path, vs Target, at the PR 2198 shapes, page 64.

Runs the source backend's ``_run_exact`` body (sglang-jax-private c948bdd,
``dsa_sparse_backend.py``) with the source modules themselves (``--source-root``:
the M4-S tar plus ``expand_cache.py`` and ``exact_attention_hlo_overlap.py``), on the
same split FP8 KV cache and the same physical slots, once per
``--dsa-attention-precision``:

* ``fp8-qk-pv``: FP8 QK and PV, split pipeline (the FP8 delivery; what Ported-S runs).
* ``fp8-qk``: FP8 QK, BF16 PV, split pipeline.
* ``bf16``: BF16 QK and PV. Prefill (>= 128 queries) first expands the FP8 cache to
  BF16 (``expand_fp8_cache_pallas``, bounded by the page table), then runs the
  HLO-overlap pipeline; decode / verify read the packed FP8 cache.

Target (``sparse_mla_attention_fused``, per-query, per-tensor k_scale=1) runs in the
same process for reference. Each timed call writes this step's KV, then attends.
Reported: device module time, top ops, error vs f64 attention on unquantized K/V.
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

FP = att.FP = 64
CASES = ("pr2198_extend_c1", "pr2198_extend_c2", "pr2198_verify_b4", "pr2198_decode_b4")
PRECISIONS = ("fp8-qk-pv", "fp8-qk", "bf16")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=list(CASES))
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--xprof-dir", required=True)
    p.add_argument("--source-root", default="/tmp/source/att")
    a = p.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    sys.path.insert(0, a.source_root)
    import jax
    import jax.numpy as jnp

    from xplane_summary import summarize
    from sgl_jax.srt.configs.dsa_config import DsaConfig
    from sgl_jax.srt.kernels.dsa.exact_attention import sparse_core_tensor_core_dsa
    from sgl_jax.srt.kernels.dsa.exact_attention_hlo_overlap import (
        sparse_core_tensor_core_dsa as sparse_core_tensor_core_dsa_hlo_overlap,
    )
    from sgl_jax.srt.kernels.dsa.expand_cache import expand_fp8_cache_pallas
    from sgl_jax.srt.kernels.dsa.fp8_kv_cache import (
        SplitFP8KVCache, cache_primary, map_cache_parts,
    )

    src = att.load_source_helpers(a.source_root)
    f8 = jnp.float8_e4m3fn

    def profile(call, sync, tag):
        d = f"{a.xprof_dir}/{tag}"
        sync()
        jax.profiler.start_trace(d)
        for _ in range(a.iters):
            call()
        sync()
        jax.profiler.stop_trace()
        x = summarize(d, a.iters, top=40)
        return {"module_us": round(sum(u for _, u in x.get("modules", [])), 1),
                "top_ops": [(n[:60], round(u, 1)) for n, u in x.get("ops", [])[:8]]}

    for name in a.cases:
        c = att.case(name)
        P, T, S = c["P"], c["T"], c["S"]
        P += P % 2  # expand_fp8_cache_pallas needs slots % 128 == 0; page P is never read
        n = P * FP
        prefill = c["mode"] == "prefill"
        rows_bf = att.bf16_rows(c)
        kv0, pe0 = jnp.asarray(c["prefix_kv"]), jnp.asarray(c["prefix_pe"])
        pad = n - kv0.shape[0]
        kv0 = jnp.pad(kv0, ((0, pad), (0, 0)))
        pe0 = jnp.pad(pe0, ((0, pad), (0, 0)))
        empty = SplitFP8KVCache(jnp.zeros((P, FP, 4, 128), jnp.uint8),
                                jnp.zeros((P, FP // 4, 4, 128), jnp.uint8))
        filled = jax.jit(lambda cc, kv_, pe_: src["_scatter_fused_kv_paged"](
            cc, kv_, pe_, jnp.array([n], jnp.int32), jnp.arange(P, dtype=jnp.int32),
            jnp.array([0, n], jnp.int32), jnp.array([0, n], jnp.int32),
            kv_lora_rank=att.LAT, fp8_dtype=f8))(empty, kv0, pe0)
        lat0, rope0 = np.asarray(filled.latent), np.asarray(filled.rope)
        slots = jnp.asarray(to_physical(c["topk"], c["pi"], c["ppr"], c["cu_q"], page=FP))
        ins = tuple(jnp.asarray(c[k]) for k in ("q_latent", "q_rope", "kv_c", "k_pe"))
        kvl, pi, cuq = (jnp.asarray(c[k]) for k in ("kv", "pi", "cu_q"))
        cukv = jnp.asarray((np.arange(S + 1) * c["ppr"] * FP).astype(np.int32))
        sm = jnp.float32(c["sm_scale"])
        row = {"case": name, "T": T, "mode": c["mode"]}

        for prec in PRECISIONS:
            plan = DsaConfig(precision=prec).resolve(
                T, eligible=True, is_prefill=prefill, fp8_kv=True)

            def step(lat, rope, ql, qr, kvc, kpe, _plan=plan):
                # dsa_sparse_backend.py _run_exact._run, physical top-k (exact mode).
                cache4d = src["_scatter_fused_kv_paged"](
                    SplitFP8KVCache(lat, rope), kvc, kpe, kvl, pi, cuq, cukv,
                    kv_lora_rank=att.LAT, fp8_dtype=f8)
                valid = (slots >= 0) & (slots < cache_primary(cache4d).shape[0] * FP)
                phys = jnp.where(valid, slots, jnp.int32(0))
                counts = jnp.sum(valid, axis=1, dtype=jnp.int32)
                bq_sparse, bq, b_topk = _plan.tiles
                padf = src["_pad_first_axis"]
                cache = map_cache_parts(lambda part: part.reshape(-1, *part.shape[2:]), cache4d)
                fp8_dtype = f8
                if _plan.cache_read_mode != "packed":
                    with jax.named_scope("ExpandFP8CacheForExtend"):
                        cache = expand_fp8_cache_pallas(
                            cache, kv_lora_rank=att.LAT, qk_rope_head_dim=att.ROPE,
                            fp8_dtype=f8, max_slot=(jnp.max(pi) + 1) * FP - 1)
                    fp8_dtype = None
                overlap = _plan.pipeline == "overlap"
                impl = sparse_core_tensor_core_dsa_hlo_overlap if overlap else sparse_core_tensor_core_dsa
                kw = {} if overlap else {"split_gather_buffers": _plan.pipeline == "split"}
                out = impl(padf(ql, bq_sparse), padf(qr, bq_sparse), cache, padf(phys, bq_sparse),
                           padf(counts, bq_sparse), sm, bq_sparse=bq_sparse, bq=bq,
                           b_topk=b_topk, fp8_dtype=fp8_dtype, fp8_q=_plan.fp8_q,
                           fp8_pv=_plan.fp8_pv, **kw)
                return out[:T], cache4d.latent, cache4d.rope

            r = {"plan": str(plan)}
            try:
                jf = jax.jit(step, donate_argnums=(0, 1))
                st = {"c": (jnp.asarray(lat0), jnp.asarray(rope0))}

                def call():
                    out, la, ro = jf(*st["c"], *ins)
                    st["c"] = (la, ro)
                    return out

                out = np.asarray(call()).astype(np.float64)
                r["accuracy_vs_bf16"] = att.error_vs_bf16(c, out, rows_bf)
                sync = lambda: jax.block_until_ready(st["c"][0])
                r["us"] = round(att._median_time(call, sync, a.iters, a.repeats)[0], 1)
                r.update(profile(call, sync, f"src_{prec}_{name}"))
            except Exception as e:  # report, do not hide; the other paths still run
                r["error"] = repr(e)[:1500]
            row[f"source_{prec}"] = r

        # Target, same rows (sequence-local), its own writer and per-query kernel, page 64.
        try:
            fill, t_step = att.target_fns(c, qblock=False)
            tn, tr = jax.jit(fill, static_argnums=(4,))(
                jnp.zeros((P, FP, 4, 128), jnp.uint8), jnp.zeros((P, FP // 4, 4, 128), jnp.uint8),
                kv0, pe0, n, jnp.arange(P, dtype=jnp.int32))
            logical = jnp.asarray(c["topk"])
            dist = jnp.asarray(np.array([S] * 3 if c["mode"] == "decode" else [0, 0, S], np.int32))
            jt = jax.jit(lambda no, ro, ql, qr, kvc, kpe: t_step(
                no, ro, ql, qr, kvc, kpe, logical, kvl, pi, cuq, dist, sm), donate_argnums=(0, 1))
            st = {"c": (tn, tr)}

            def tcall():
                out, no, ro = jt(*st["c"], *ins)
                st["c"] = (no, ro)
                return out

            out = np.asarray(tcall()).astype(np.float64)
            r = {"accuracy_vs_bf16": att.error_vs_bf16(c, out, rows_bf)}
            tsync = lambda: jax.block_until_ready(st["c"][0])
            r["us"] = round(att._median_time(tcall, tsync, a.iters, a.repeats)[0], 1)
            r.update(profile(tcall, tsync, f"target_{name}"))
        except Exception as e:
            r = {"error": repr(e)[:1500]}
        row["target"] = r
        print("SRCPATH " + json.dumps(row), flush=True)
    print("SRCPATH_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
