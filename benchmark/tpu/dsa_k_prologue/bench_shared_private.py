"""Test only: shared + private top-k split for the extend sparse attention.

Idea: in a group of ``g`` consecutive queries, the rows every query picked
(the intersection) are gathered once and attended by all ``g`` queries; each
query's remaining rows are gathered and attended as before; the two partial
results are merged with their log-sum-exps. Gathered rows drop from
``g * 2048`` to ``Ks + g * (2048 - Ks)`` per group with no extra attention work.

The prototype reuses ``sparse_ragged_paged_attention`` unchanged:

* shared part: MLA's latent K/V is shared across heads, so a group's ``g``
  queries are just ``g * H`` heads of one token. ``q`` reshapes to
  ``[T/g, g*H, D]`` and the shared rows are that token's top-k (``Ks`` each).
* private part: ``[T, 2048 - Ks]`` rows per query, the usual call.
* both with ``return_lse``; merged in f32.

``Ks`` must be static: one value per call, the smallest intersection over its
groups rounded down to 64 (so every query's private set has exactly
``2048 - Ks`` rows and nothing is padded). The intersections are built on the
host from the real GLM-5.2 top-k dump (exp-fj8ahrq5pz ``raw-*.npy``, logical
positions < 36864); their device cost is not measured here.

Same extend_c1 cache (SparseCore layout, physical slots) as the sweep; the
dump's 2048 x 2048 rows replace the synthetic top-k. Prints ``SPL`` lines.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_fp8_attention_ab as att  # noqa: E402

FP = att.FP = 64
CASE = "pr2198_extend_c1"
CHUNK, BATCH = 128, 16
ATOM = 256  # Ks rounding: keeps 2048 - Ks a multiple of 256 (private gather atoms 16)
OWNERS, OCC = 32, 64  # native SC gather: subcores per device, rows per atom


def atoms_for(tokens, topk):
    """Largest atoms_per_batch <= 16 the native SC gather accepts for this chunk."""
    out = tokens * topk
    return max(a for a in range(1, 17) if out % (OWNERS * a * OCC) == 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--files", default="raw-0096-layer0,raw-0097-layer10,raw-0099-layer40")
    p.add_argument("--groups", default="2,4")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--xprof-dir", required=True)
    a = p.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    from xplane_summary import summarize
    from sglang_tpu.attention.pallas.mla_sparse import kv_cache_utils as kcu
    from sglang_tpu.attention.pallas.mla_sparse.kernel import sparse_ragged_paged_attention

    f8 = jnp.float8_e4m3fn
    fmax = float(jnp.finfo(f8).max)
    q8 = lambda x: jnp.clip(x.astype(jnp.float32), -fmax, fmax).astype(f8)
    lay = kcu.KVCacheLayout.SPARSECORE
    c = att.case(CASE)
    P, T = c["P"], c["T"]
    n = P * FP
    ns = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.NOPE, lay, P, FP, 512)
    rs = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.ROPE, lay, P, FP, 64)
    fill = jax.jit(lambda nope, rope, kv_, pe_: kcu.update_sparse_mla_kv_cache_jax(
        nope, rope, kv_, pe_, jnp.array([n], jnp.int32), jnp.arange(P, dtype=jnp.int32),
        jnp.array([0, n], jnp.int32), nope_spec=ns, rope_spec=rs))
    nope, rope = fill(jnp.zeros(ns.shape, ns.jax_dtype), jnp.zeros(rs.shape, rs.jax_dtype),
                      q8(jnp.asarray(c["prefix_kv"])), q8(jnp.asarray(c["prefix_pe"])))
    q = jnp.concatenate([jnp.asarray(c["q_latent"]), jnp.asarray(c["q_rope"])], axis=-1)
    H = q.shape[1]
    sm = float(c["sm_scale"])
    pi = np.asarray(c["pi"])  # one sequence: its page table is pi[:ppr]
    phys = lambda lg: np.where(lg >= 0, pi[np.maximum(lg, 0) // FP] * FP
                               + np.maximum(lg, 0) % FP, -1).astype(np.int32)

    def attend(q_, rows, chunk, batch, lse):
        t = rows.shape[0]
        return sparse_ragged_paged_attention(
            q_, nope, rope, rows, None, jnp.array([0, t], jnp.int32),
            jnp.array([0, 0, 1], jnp.int32), sm_scale=sm, k_scale=1.0,
            cache_layout="sparsecore", gather_and_attention_chunk_size=chunk,
            attention_kernel_batch_size=batch,
            sparsecore_atoms_per_batch=atoms_for(chunk, rows.shape[1]),
            return_lse=lse)

    def timed(tag, fn):
        sync = lambda: jax.block_until_ready(st["o"])
        st = {}

        def call():
            st["o"] = fn()
            return st["o"]

        med, runs = att._median_time(call, sync, a.iters, a.repeats)
        d = f"{a.xprof_dir}/{tag}"
        jax.profiler.start_trace(d)
        for _ in range(a.iters):
            call()
        sync()
        jax.profiler.stop_trace()
        x = summarize(d, a.iters, top=6)
        return dict(us=round(med, 1), us_range=[round(min(runs), 1), round(max(runs), 1)],
                    module_us=round(sum(u for _, u in x.get("modules", [])), 1),
                    top_ops=[(nm[:40], round(u, 1)) for nm, u in x.get("ops", [])[:6]])

    paths = {os.path.basename(f)[:-4]: f
             for f in glob.glob(f"{a.raw_dir}/**/raw-*.npy", recursive=True)}
    for name in a.files.split(","):
        row = {"file": name}
        try:
            raw = np.load(paths[name]).astype(np.int32)
            assert raw.shape == (T, 2048) and (raw >= 0).all(), raw.shape
            c["topk"] = raw  # error_vs_bf16 reads the same rows
            rows_bf = att.bf16_rows(c)
            full = phys(raw)
            full = jnp.asarray(full)
            base_fn = jax.jit(lambda q_, r_: attend(q_, r_, CHUNK, BATCH, False))
            ref = np.asarray(base_fn(q, full))[..., :512].astype(np.float32)
            row["base_err"] = att.error_vs_bf16(c, ref.astype(np.float64), rows_bf)
            row["base"] = timed(f"{name}_base", lambda: base_fn(q, full))
            print("SPL " + json.dumps(row), flush=True)
        except Exception as e:  # report, do not hide
            import traceback

            traceback.print_exc()
            print("SPL " + json.dumps({"file": name, "fail": repr(e)[:500]}), flush=True)
            continue
        for g in (int(x) for x in a.groups.split(",")):
            r = {"file": name, "g": g}
            try:
                sets = [np.sort(x) for x in raw]
                inter = []
                for b in range(0, T, g):
                    s = sets[b]
                    for t in sets[b + 1:b + g]:
                        s = np.intersect1d(s, t, assume_unique=True)
                    inter.append(s)
                sizes = np.array([len(s) for s in inter])
                ks = int(sizes.min()) // ATOM * ATOM
                kp = 2048 - ks
                r.update(ks=ks, inter_min=int(sizes.min()), inter_mean=round(float(sizes.mean()), 1),
                         rows_frac=round((T // g * ks + T * kp) / (T * 2048), 4),
                         rows_frac_if_mean=round(1 - (g - 1) / g * sizes.mean() / 2048, 4))
                if ks == 0:
                    r["skip"] = "empty intersection in some group"
                    print("SPL " + json.dumps(r), flush=True)
                    continue
                sh = np.stack([s[:ks] for s in inter])  # [T/g, Ks]
                pv = np.stack([np.setdiff1d(sets[t], sh[t // g], assume_unique=True)
                               for t in range(T)])  # [T, Kp]
                assert pv.shape == (T, kp)
                sh_p, pv_p = jnp.asarray(phys(sh)), jnp.asarray(phys(pv))
                r["atoms"] = [atoms_for(CHUNK // g, ks), atoms_for(CHUNK, kp)]

                def split(q_, sh_p, pv_p, g=g):
                    o1, l1 = attend(q_.reshape(T // g, g * H, q_.shape[-1]), sh_p,
                                    CHUNK // g, max(1, BATCH // g), True)
                    o2, l2 = attend(q_, pv_p, CHUNK, BATCH, True)
                    o1 = o1.reshape(T, H, -1)[..., :512].astype(jnp.float32)
                    l1 = l1.reshape(T, H)[..., None]
                    o2 = o2[..., :512].astype(jnp.float32)
                    l2 = l2[..., None]
                    m = jnp.maximum(l1, l2)
                    w1, w2 = jnp.exp(l1 - m), jnp.exp(l2 - m)
                    return ((o1 * w1 + o2 * w2) / (w1 + w2)).astype(q_.dtype)

                sj = jax.jit(split, static_argnames=("g",))
                out = np.asarray(sj(q, sh_p, pv_p)).astype(np.float32)
                den = np.maximum(np.abs(ref), 1e-3)
                r["vs_base"] = dict(max_abs=float(np.abs(out - ref).max()),
                                    p99_rel=float(np.percentile(np.abs(out - ref) / den, 99)))
                r["err"] = att.error_vs_bf16(c, out.astype(np.float64), rows_bf)
                r.update(timed(f"{name}_g{g}", lambda: sj(q, sh_p, pv_p)))
            except Exception as e:  # report, do not hide
                import traceback

                traceback.print_exc()
                r["fail"] = repr(e)[:500]
            print("SPL " + json.dumps(r), flush=True)
            print("SPLT " + json.dumps({k: r.get(k) for k in (
                "file", "g", "ks", "inter_mean", "rows_frac", "atoms", "us", "module_us", "fail")}), flush=True)
        print("SPLT " + json.dumps({"file": name, "base_us": row["base"]["us"],
                                    "base_module_us": row["base"]["module_us"]}), flush=True)
    print("SPL_DONE", flush=True)


if __name__ == "__main__":
    main()
