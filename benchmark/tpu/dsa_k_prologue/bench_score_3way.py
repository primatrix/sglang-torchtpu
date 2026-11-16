"""Index scoring + radix top-k, source contract (BF16 cache, page 64): Source / Ported-S / Target.

* Source: ``sgl_jax.srt.kernels.dsa.indexer.compute_scores_and_select_topk_indices``
  at the pinned commit, unpacked from a tarball by the manifest (never committed).
* ported_s_jax: the copied ``source_dsa`` indexer, called from JAX.
* ported_s: the ``dsa_index_score_topk_bf16`` torch op, compiled (stage torch).
* target: ``dsa_index_topk._select_topk`` on the FP8 UE8M0 record cache, page 128,
  same keys without Hadamard. Different format; reported apart.

Radix top-k is unordered, so rows are compared after sorting. Every side is
also scored against exact f64 scores of its own keys (selection shortfall).
``--stage torch`` then ``--stage jax``.
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PAGE, D, HEADS, K = 64, 128, 32, 2048

# name -> (q_lens, prefixes, one_token_per_seq, output_physical_slots)
CASES = {
    "decode_b8": ([1] * 8, [8191 + 37 * s for s in range(8)], True, True),
    "decode_b64": ([1] * 64, [8191 + 37 * s for s in range(64)], True, True),
    "extend_2048": ([2048], [8192], False, True),
    "extend_8192": ([2048] * 4, [8192] * 4, False, True),
    "extend_2048_logical": ([2048], [8192], False, False),
    "packed_2x6": ([6, 6], [8192, 8000], False, True),
}


def case(name, seed=0):
    q_lens, prefixes, decode, phys = CASES[name]
    r = np.random.default_rng(seed + sum(q_lens))
    seqs = len(q_lens)
    kv = np.add(prefixes, q_lens)
    pps = int(-(-kv.max() // PAGE))
    pps = -(-pps // 2) * 2  # max_kv a multiple of 128
    # Request s owns pages 1 + s*pps ... in reverse order (a wrong lookup lands elsewhere).
    table = np.concatenate([np.arange(1 + (s + 1) * pps - 1, s * pps, -1) for s in range(seqs)])
    T = int(sum(q_lens))
    return dict(
        name=name, decode=decode, phys=phys, pps=pps, T=T,
        q=r.normal(0, 1, (T, HEADS, D)).astype(np.float32),
        w=r.uniform(0.1, 1.0, (T, HEADS)).astype(np.float32),
        cache=r.normal(0, 1, (1 + seqs * pps, PAGE, D)).astype(np.float32),
        seq_lens=kv.astype(np.int32), page_indices=table.astype(np.int32),
        cu_q=np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32),
        cu_kv=(np.arange(seqs + 1) * pps * PAGE).astype(np.int32),
        dist=np.array([seqs] * 3 if decode else [0, 0, seqs], np.int32),
    )


def _bf16(a):
    import ml_dtypes

    return np.asarray(a, np.float32).astype(ml_dtypes.bfloat16)


def _sorted(rows):
    return np.sort(np.asarray(rows), axis=1)


def _median_time(call, sync, iters, repeats):
    call()
    sync()
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            call()
        sync()
        runs.append((time.perf_counter() - t0) / iters * 1e6)
    return statistics.median(runs), runs


def _positions(c, rows):
    """Per-query logical positions from top-k rows (physical slots resolved back)."""
    rows = np.asarray(rows)
    if not c["phys"]:
        return rows
    seq = np.searchsorted(c["cu_q"][1:], np.arange(c["T"]), side="right")
    out = np.full_like(rows, -1)
    for t in range(c["T"]):
        tab = c["page_indices"][seq[t] * c["pps"]:(seq[t] + 1) * c["pps"]]
        inv = {int(p): i for i, p in enumerate(tab)}
        for j, s in enumerate(rows[t]):
            if s >= 0:
                out[t, j] = inv[int(s) // PAGE] * PAGE + int(s) % PAGE
    return out


def shortfall(c, rows, queries=64):
    """Per sampled query: 1 - (lowest exact score picked) / (exact k-th best), f64."""
    pos = _positions(c, rows)
    seq = np.searchsorted(c["cu_q"][1:], np.arange(c["T"]), side="right")
    keys = _bf16(c["cache"]).astype(np.float64)
    q = _bf16(c["q"]).astype(np.float64)
    vals = []
    for t in np.linspace(0, c["T"] - 1, min(queries, c["T"])).astype(int):
        s = seq[t]
        q_len = c["cu_q"][s + 1] - c["cu_q"][s]
        n = c["seq_lens"][s] - q_len + (t - c["cu_q"][s]) + 1
        tab = c["page_indices"][s * c["pps"]:(s + 1) * c["pps"]]
        kk = keys[tab].reshape(-1, D)[:n]
        sc = (np.maximum(q[t] @ kk.T, 0) * c["w"][t][:, None]).sum(0)
        picked = pos[t][pos[t] >= 0]
        if n <= K:
            vals.append(0.0 if len(picked) == n else 1.0)
            continue
        kth = np.partition(-sc, K - 1)[K - 1] * -1
        vals.append(max(0.0, 1 - sc[picked].min() / kth) if len(picked) == K else 1.0)
    v = np.asarray(vals)
    return {"max": float(v.max()), "p99": float(np.quantile(v, 0.99)),
            "frac_over_2pct": float(np.mean(v > 0.02))}


def stage_torch(a):
    os.environ["JAX_PLATFORMS"] = "cpu"  # TorchTPU owns the device
    import torch
    import torch_tpu  # noqa: F401

    from bench_torch_ab import _profile
    from sglang_tpu.dsa_bf16_index_score_op import build_dsa_index_score_topk_bf16_op

    op = build_dsa_index_score_topk_bf16_op()
    sync = torch.accelerator.synchronize
    rows = {}
    for name in a.cases:
        c = case(name)
        bf = lambda x: torch.from_numpy(_bf16(x).view(np.int16)).view(torch.bfloat16).to("tpu")
        i32 = lambda x: torch.from_numpy(x).to("tpu")
        args = (bf(c["q"]), torch.from_numpy(c["w"]).to("tpu"), bf(c["cache"]),
                i32(c["seq_lens"]), i32(c["page_indices"]), i32(c["cu_q"]),
                i32(c["cu_kv"]), i32(c["dist"]))
        static = (K, c["pps"], c["decode"], c["phys"])
        fn = torch.compile(lambda *x: op(*x, *static), backend="tpu", fullgraph=True, dynamic=False)
        out = fn(*args).cpu().numpy()
        sync()
        np.save(os.path.join(a.npz_dir, f"score_torch_{name}.npy"), out)
        row = {"case": name}
        row["us"], row["runs"] = _median_time(lambda: fn(*args), sync, a.iters, a.repeats)
        if a.xprof_dir:
            row["xprof"] = _profile(fn, args, a.iters, f"{a.xprof_dir}/score_torch_{name}")
        rows[name] = row
        print("STORCH " + json.dumps({k: v for k, v in row.items() if k != "xprof"}), flush=True)
    with open(os.path.join(a.npz_dir, "score_torch_rows.json"), "w") as f:
        json.dump(rows, f, indent=1)


def _target(c):
    """Target top-k on FP8 records of the same (un-Hadamarded) keys, page 128."""
    import jax.numpy as jnp

    from sglang_tpu.attention.pallas import dsa_index_topk as dit
    from sglang_tpu.attention.pallas.util import quantize_fp8_ue8m0

    # Same logical layout at page 128: request s's tokens in pages 1 + s*pps128 ...
    seqs = len(c["seq_lens"])
    pps = -(-int(c["seq_lens"].max()) // 128)
    pages = (1 + np.arange(seqs * pps)).astype(np.int32)
    keys = np.zeros((seqs * pps * 128, D), np.float32)
    for s in range(seqs):
        tab = c["page_indices"][s * c["pps"]:(s + 1) * c["pps"]]
        k = c["cache"][tab].reshape(-1, D)[: c["seq_lens"][s]]
        keys[s * pps * 128: s * pps * 128 + len(k)] = k
    kf, _, kb = quantize_fp8_ue8m0(jnp.asarray(_bf16(keys)))
    cache = jnp.zeros(dit.index_key_cache_shape(1 + seqs * pps, 128, D), jnp.uint8)
    cache = dit._store_keys(cache, kf, kb, jnp.full((1,), seqs * pps * 128, jnp.int32),
                            jnp.asarray(pages), jnp.array([0, seqs * pps * 128], jnp.int32),
                            jnp.array([0, 0, 1], jnp.int32))
    meta = (jnp.asarray(c["seq_lens"]), jnp.asarray(pages), jnp.asarray(c["cu_q"]),
            jnp.asarray(c["dist"]))

    def fn(q, w, cache):
        return dit._select_topk(q, w, cache, *meta, None, weight_scale=1.0, topk=K,
                                decode_query_size=1, segments="all", row_count=0)[0]

    return fn, (jnp.asarray(_bf16(c["q"])), jnp.asarray(c["w"])), cache


def stage_jax(a):
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    from xplane_summary import summarize
    from sglang_tpu.attention.pallas.source_dsa.kernels.dsa import indexer as ported

    sys.path.insert(0, a.source_root)
    from sgl_jax.srt.kernels.dsa import indexer as source  # noqa: E402

    with open(os.path.join(a.npz_dir, "score_torch_rows.json")) as f:
        torch_rows = json.load(f)

    def run(fn, args, tag):
        f = jax.jit(fn)
        out = f(*args)
        sync = lambda: jax.block_until_ready(out)
        med, runs = _median_time(lambda: f(*args), lambda: jax.block_until_ready(f(*args)),
                                 a.iters, a.repeats)
        res = {"us": med, "runs": runs}
        if a.xprof_dir:
            d = f"{a.xprof_dir}/score_jax_{tag}"
            sync()
            jax.profiler.start_trace(d)
            for _ in range(a.iters):
                o = f(*args)
            jax.block_until_ready(o)
            jax.profiler.stop_trace()
            try:
                res["xprof"] = summarize(d, a.iters)
            except Exception as e:  # report, do not hide
                res["xprof"] = {"error": repr(e)[:2000]}
        return np.asarray(out), res

    report = {"jax": jax.__version__, "device": str(jax.devices()[0]), "rows": []}
    for name in a.cases:
        c = case(name)
        args = (jnp.asarray(_bf16(c["q"])), jnp.asarray(c["w"]), jnp.asarray(_bf16(c["cache"])),
                *(jnp.asarray(c[k]) for k in ("seq_lens", "page_indices", "cu_q", "cu_kv", "dist")))
        kw = dict(k=K, pages_per_seq=c["pps"], one_token_per_seq=c["decode"],
                  topk_impl="radix", output_physical_slots=c["phys"])
        row = {"case": name, "T": c["T"], "physical": c["phys"]}
        outs = {}
        for side, mod in (("source", source), ("ported_s_jax", ported)):
            try:
                out, res = run(lambda *x, m=mod: m.compute_scores_and_select_topk_indices(*x, **kw),
                               args, f"{name}_{side}")
                outs[side] = out
                row[side] = {**res, "selection": shortfall(c, out)}
            except Exception as e:  # report, do not hide
                row[side] = {"error": repr(e)[:3000]}
        t_out = np.load(os.path.join(a.npz_dir, f"score_torch_{name}.npy"))
        row["ported_s"] = {**torch_rows[name], "selection": shortfall(c, t_out)}
        if "source" in outs:
            src = _sorted(outs["source"])
            if "ported_s_jax" in outs:
                row["ported_s_jax_vs_source_exact"] = bool(np.array_equal(_sorted(outs["ported_s_jax"]), src))
            row["ported_s_vs_source_exact"] = bool(np.array_equal(_sorted(t_out), src))
        try:
            fn, targs, cache = _target(c)
            out, res = run(lambda q, w, cc: fn(q, w, cc), (*targs, cache), f"{name}_target")
            tc = dict(c, phys=False)
            row["target"] = {**res, "selection": shortfall(tc, out),
                             "note": "FP8 UE8M0 records, page 128, no Hadamard"}
        except Exception as e:  # report, do not hide
            row["target"] = {"error": repr(e)[:3000]}
        print("SCORE " + json.dumps(_summary(row)), flush=True)
        report["rows"].append(row)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)


def _summary(row):
    out = {"case": row["case"], "T": row["T"]}
    for side in ("source", "ported_s_jax", "ported_s", "target"):
        r = row.get(side, {})
        if "error" in r:
            out[side] = {"error": r["error"][:400]}
            continue
        x = r.get("xprof", {})
        out[side] = {
            "wall_us": r.get("us"),
            "module_us": round(sum(u for _, u in x.get("modules", [])), 1) if "modules" in x else None,
            "top_ops": [(n[:50], round(u, 1)) for n, u in x.get("ops", [])[:5]],
            "sel": r.get("selection"),
        }
    for k in ("ported_s_jax_vs_source_exact", "ported_s_vs_source_exact"):
        out[k] = row.get(k)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("torch", "jax"), required=True)
    p.add_argument("--cases", nargs="+", default=list(CASES))
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--npz-dir", default="/tmp/score3")
    p.add_argument("--xprof-dir", default="")
    p.add_argument("--source-root", default="/tmp/source/m3s")
    p.add_argument("--out", default="/tmp/score3/report.json")
    a = p.parse_args()
    os.makedirs(a.npz_dir, exist_ok=True)
    stage_torch(a) if a.stage == "torch" else stage_jax(a)


if __name__ == "__main__":
    main()
