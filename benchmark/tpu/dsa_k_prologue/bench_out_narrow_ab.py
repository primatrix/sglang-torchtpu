"""Test only: does a 512-wide attention output remove the ~83 us output slice?

bench_small_ops_timeline (exp-9pxwfo32rg) found ``slice.92`` (bf16[2048,64,640]
-> [..., :512], ~83 us) at the end of the extend_c1 call, after the last
gather. The kernel's output aliases the padded q (640 wide), so the NoPE part
is sliced out afterwards. ``narrow_output=True`` (test-only kernel flag) makes
the kernel write a separate 512-wide output (first chunk a fresh buffer, later
chunks aliased with it), so no slice is needed.

Production-shaped step (row-major ql_nope / q_pe concatenated in the jit,
SparseCore layout, physical slots, chunk 128 / batch 16 / atoms 16 /
overlap_rope), two variants on identical inputs: ``out_slice`` (as today) and
``out_narrow``. Per variant: 640-wide slices left in the HLO, median wall time,
device module time, top ops, and bit equality with ``out_slice``. Prints ``ONB``
lines (``ONBT``: short timing lines).
"""

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
    xp = sys.argv[1] if len(sys.argv) > 1 else "/tmp/out_narrow"
    iters, repeats = 10, 5
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
    fill = jax.jit(lambda nope, rope, kv_, pe_: kcu.update_sparse_mla_kv_cache_jax(
        nope, rope, kv_, pe_, jnp.array([n], jnp.int32), jnp.arange(P, dtype=jnp.int32),
        jnp.array([0, n], jnp.int32), nope_spec=ns, rope_spec=rs))
    nope0, rope0 = fill(jnp.zeros(ns.shape, ns.jax_dtype), jnp.zeros(rs.shape, rs.jax_dtype),
                        q8(jnp.asarray(c["prefix_kv"])), q8(jnp.asarray(c["prefix_pe"])))
    caches0 = (np.asarray(nope0), np.asarray(rope0))
    kvc, kpe = q8(jnp.asarray(c["kv_c"])), q8(jnp.asarray(c["k_pe"]))
    kvl, pi, cuq = (jnp.asarray(c[k]) for k in ("kv", "pi", "cu_q"))
    dist = jnp.asarray([0, 0, S], jnp.int32)
    rows = jnp.asarray(to_physical(c["topk"], c["pi"], c["ppr"], c["cu_q"], page=FP))
    sm = float(c["sm_scale"])
    rows_bf = att.bf16_rows(c)

    def make_step(narrow):
        def step(nope_, rope_, ql_, qpe_, kvc_, kpe_):
            active = jnp.minimum(cuq, cuq[dist[2]])
            nope_, rope_ = kcu.update_sparse_mla_kv_cache(
                nope_, rope_, kvc_, kpe_, kvl, pi, active, nope_spec=ns, rope_spec=rs)
            out = sparse_ragged_paged_attention(
                jnp.concatenate([ql_, qpe_], axis=-1), nope_, rope_, rows, None, cuq, dist,
                sm_scale=sm, k_scale=1.0, cache_layout="sparsecore", narrow_output=narrow,
                **KW)
            return out[..., :512], nope_, rope_
        return step

    sh = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    fmt = Format(Layout(major_to_minor=(0, 1, 2)), sh)
    qa = jax.device_put(jnp.asarray(c["q_latent"]), fmt)
    qb = jax.device_put(jnp.asarray(c["q_rope"]), fmt)
    ref = None
    for tag, narrow in (("out_slice", False), ("out_narrow", True)):
        row = {"variant": tag}
        try:
            jf = jax.jit(make_step(narrow), donate_argnums=(0, 1),
                         in_shardings=(sh, sh, fmt, fmt, sh, sh))
            st = {"c": tuple(jnp.asarray(x) for x in caches0)}
            hlo = jf.lower(*st["c"], qa, qb, kvc, kpe).compile().as_text()
            sl = [ln.strip()[:160] for ln in hlo.splitlines()
                  if re.search(r"bf16\[2048,64,512\]\S* (slice|fusion)\(bf16\[2048,64,640\]", ln)]
            row["slice640_ops"] = len(sl)
            row["slice640_lines"] = sl[:2]

            def call():
                out, no, ro = jf(*st["c"], qa, qb, kvc, kpe)
                st["c"] = (no, ro)
                return out

            out = np.asarray(call())
            if ref is None:
                ref = out
            row["exact_vs_slice"] = bool(np.array_equal(out.view(np.uint16), ref.view(np.uint16)))
            row["finite"] = bool(np.isfinite(out.astype(np.float32)).all())
            row["error"] = att.error_vs_bf16(c, out.astype(np.float64), rows_bf)
            sync = lambda: jax.block_until_ready(st["c"][0])
            med, runs = att._median_time(call, sync, iters, repeats)
            row["us"] = round(med, 1)
            row["us_range"] = [round(min(runs), 1), round(max(runs), 1)]
            d = f"{xp}/onb_{tag}"
            sync()
            jax.profiler.start_trace(d)
            for _ in range(iters):
                call()
            sync()
            jax.profiler.stop_trace()
            x = summarize(d, iters, top=8)
            row["module_us"] = round(sum(u for _, u in x.get("modules", [])), 1)
            row["top_ops"] = [(nm[:50], round(u, 1)) for nm, u in x.get("ops", [])[:6]]
        except Exception as e:  # report, do not hide
            import traceback

            traceback.print_exc()
            row["fail"] = repr(e)[:600]
        print("ONB " + json.dumps(row), flush=True)
        print("ONBT " + json.dumps({k: row.get(k) for k in (
            "variant", "slice640_ops", "exact_vs_slice", "finite", "us", "us_range",
            "module_us", "fail")}), flush=True)
    print("ONB_DONE", flush=True)


if __name__ == "__main__":
    main()
