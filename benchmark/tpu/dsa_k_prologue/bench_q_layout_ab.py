"""Test only: does pinning q to row-major remove the 144 MB q relayout copy?

bench_copy_hlo (exp-vvmc2ippfj) found the ~120 us SparseCore
``OFFLOAD_DATA_FORMATTING`` copy before the first gather is ``copy(q_)``:
bf16[2048,64,576] from layout {0,2,1} (the jit parameter's) to {2,1,0}
(what the attention kernel reads). Same extend_c1 step as the sweep
(SparseCore layout, physical slots, chunk 128 / batch 16 / atoms 16 /
overlap_rope), two variants on identical inputs:

* ``q_default``: q passed as before (XLA picks the parameter layout).
* ``q_rowmajor``: q committed and pinned to major_to_minor (0, 1, 2) through
  ``jax.jit(in_shardings=Format(Layout(...), sharding))``.

Per variant: the q-shaped copies left in the compiled HLO, the parameter
layout of q, median wall time, device module time, and whether the output
equals ``q_default``'s bit for bit. Prints ``QAB`` lines.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_fp8_attention_ab as att  # noqa: E402
from check_physical_topk import to_physical  # noqa: E402

FP = att.FP = 64
CASE = "pr2198_extend_c1"
KW = dict(gather_and_attention_chunk_size=128, attention_kernel_batch_size=16,
          sparsecore_atoms_per_batch=16, sparsecore_gather_wait_mode="overlap_rope")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--xprof-dir", required=True)
    a = p.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp
    from jax.experimental.layout import Format, Layout

    from xplane_summary import summarize
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
    kv0, pe0 = q8(jnp.asarray(c["prefix_kv"])), q8(jnp.asarray(c["prefix_pe"]))
    fill = jax.jit(lambda nope, rope, kv_, pe_: kcu.update_sparse_mla_kv_cache_jax(
        nope, rope, kv_, pe_, jnp.array([n], jnp.int32), jnp.arange(P, dtype=jnp.int32),
        jnp.array([0, n], jnp.int32), nope_spec=ns, rope_spec=rs))
    nope0, rope0 = fill(jnp.zeros(ns.shape, ns.jax_dtype), jnp.zeros(rs.shape, rs.jax_dtype),
                        kv0, pe0)
    caches0 = (np.asarray(nope0), np.asarray(rope0))
    q = jnp.concatenate([jnp.asarray(c["q_latent"]), jnp.asarray(c["q_rope"])], axis=-1)
    kvc, kpe = q8(jnp.asarray(c["kv_c"])), q8(jnp.asarray(c["k_pe"]))
    kvl, pi, cuq = (jnp.asarray(c[k]) for k in ("kv", "pi", "cu_q"))
    dist = jnp.asarray(np.array([0, 0, S], np.int32))
    rows = jnp.asarray(to_physical(c["topk"], c["pi"], c["ppr"], c["cu_q"], page=FP))
    sm = float(c["sm_scale"])
    rows_bf = att.bf16_rows(c)

    def step(nope, rope, q_, kvc_, kpe_):
        active = jnp.minimum(cuq, cuq[dist[2]])
        nope, rope = kcu.update_sparse_mla_kv_cache(
            nope, rope, kvc_, kpe_, kvl, pi, active, nope_spec=ns, rope_spec=rs)
        out = sparse_ragged_paged_attention(
            q_, nope, rope, rows, None, cuq, dist, sm_scale=sm, k_scale=1.0,
            cache_layout="sparsecore", **KW)
        return out[..., :512], nope, rope

    def step_split(nope, rope, ql_, qpe_, kvc_, kpe_):
        # Production (mla_sparse_fused): two dense row-major torch buffers,
        # concatenated inside the JAX function.
        return step(nope, rope, jnp.concatenate([ql_, qpe_], axis=-1), kvc_, kpe_)

    dev = jax.devices()[0]
    sh = jax.sharding.SingleDeviceSharding(dev)
    ref = None
    ql_in, qpe_in = jnp.asarray(c["q_latent"]), jnp.asarray(c["q_rope"])
    for tag in ("q_default", "q_rowmajor", "q_split"):
        row = {"variant": tag}
        try:
            if tag == "q_default":
                jf = jax.jit(step, donate_argnums=(0, 1))
                qin = q
            elif tag == "q_rowmajor":
                fmt = Format(Layout(major_to_minor=(0, 1, 2)), sh)
                jf = jax.jit(step, donate_argnums=(0, 1),
                             in_shardings=(sh, sh, fmt, sh, sh))
                qin = jax.device_put(q, fmt)
            else:
                fmt = Format(Layout(major_to_minor=(0, 1, 2)), sh)
                sj = jax.jit(step_split, donate_argnums=(0, 1),
                             in_shardings=(sh, sh, fmt, fmt, sh, sh))
                qa, qb = jax.device_put(ql_in, fmt), jax.device_put(qpe_in, fmt)
                jf = lambda n_, r_, _q, k1, k2: sj(n_, r_, qa, qb, k1, k2)
                jf.lower = lambda n_, r_, _q, k1, k2: sj.lower(n_, r_, qa, qb, k1, k2)
                qin = None
            st = {"c": tuple(jnp.asarray(x) for x in caches0)}
            hlo = jf.lower(*st["c"], qin, kvc, kpe).compile().as_text()
            qcopies = [ln.strip()[:160] for ln in hlo.splitlines()
                       if re.search(r"= bf16\[2048,64,576\]\{[^}]*\} copy\(", ln)]
            entry = [ln.strip()[:160] for ln in hlo.splitlines()
                     if "ENTRY" in ln][:1]
            row["q_copies"] = len(qcopies)
            row["q_concat"] = [ln.strip()[:200] for ln in hlo.splitlines()
                               if re.search(r"= bf16\[2048,64,576\]\{[^}]*\} (concatenate|fusion)\(", ln)][:3]
            row["q_copy_lines"] = qcopies[:3]
            row["entry"] = entry
            try:
                row["q_in_layout"] = str(jf.lower(*st["c"], qin, kvc, kpe).compile()
                                         .input_formats[0][2])[:200]
            except Exception as e:  # report, do not hide
                row["q_in_layout"] = f"? {e!r}"[:200]

            def call():
                out, no, ro = jf(*st["c"], qin, kvc, kpe)
                st["c"] = (no, ro)
                return out

            out = np.asarray(call())
            if ref is None:
                ref = out
            row["exact_vs_default"] = bool(np.array_equal(out.view(np.uint16),
                                                          ref.view(np.uint16)))
            row["error"] = att.error_vs_bf16(c, out.astype(np.float64), rows_bf)
            sync = lambda: jax.block_until_ready(st["c"][0])
            med, runs = att._median_time(call, sync, a.iters, a.repeats)
            row["us"] = round(med, 1)
            row["us_range"] = [round(min(runs), 1), round(max(runs), 1)]
            d = f"{a.xprof_dir}/qab_{tag}"
            sync()
            jax.profiler.start_trace(d)
            for _ in range(a.iters):
                call()
            sync()
            jax.profiler.stop_trace()
            x = summarize(d, a.iters, top=8)
            row["module_us"] = round(sum(u for _, u in x.get("modules", [])), 1)
            row["top_ops"] = [(nm[:60], round(u, 1)) for nm, u in x.get("ops", [])[:8]]
        except Exception as e:  # report, do not hide
            import traceback

            row["fail"] = repr(e)[:800]
            traceback.print_exc()
        print("QAB " + json.dumps(row), flush=True)
        print("QABT " + json.dumps({k: row.get(k) for k in ("variant", "q_copies", "q_concat", "exact_vs_default", "us", "us_range", "module_us", "fail")}), flush=True)
    print("QAB_DONE", flush=True)


if __name__ == "__main__":
    main()
