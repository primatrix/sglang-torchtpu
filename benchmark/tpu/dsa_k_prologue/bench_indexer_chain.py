"""Full DSA indexer (S1-S4), source contract: Source / Ported-S (JAX, torch op) / Target.

* Source: the pinned source kernels at page 64 (``--source-k``, ``--source-q``,
  ``--source-root`` for the scorer package), chained by hand: fused K store
  (aligned extend; ``token_slots`` for decode), fused Q tail (prefill; the
  source's ``apply_rotary_emb`` + einsum otherwise), then
  ``compute_scores_and_select_topk_indices(topk_impl="radix",
  output_physical_slots=False)``.
* ported_s_jax: ``dsa_bf16_indexer.bf16_indexer_step`` on the framework's page-128
  page table (framework page p = source pages 2p, 2p+1).
* ported_s: the ``dsa_bf16_indexer`` torch op, compiled.
* Target: the target's FP8 indexer step (``dsa_index_topk.dsa_indexer_step``) on
  keys from ``dsa_key_prologue.key_chain_xla`` (W8A8, eps 1e-6, no Hadamard),
  FP8 UE8M0 records at page 128. Different format and precision: labelled.

Source and Ported-S get the same inputs and must store byte-identical caches
and pick identical top-k sets. Selection is scored per side against f64 exact
scores of that side's own stored keys and queries (Target: also against its own
FP8-dequantized keys). ``--stage torch`` then ``--stage jax``.
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

HIDDEN, D, HEADS, K, FP = 6144, 128, 32, 2048, 128

# name -> (q_lens, prefixes, mode); mode: extend | decode | verify (False / True = extend / decode)
CASES = {
    "extend_2048": ([2048], [8192], False),
    "extend_4x2048": ([2048] * 4, [8192] * 4, False),
    "extend_unaligned": ([2000], [8200], False),
    "decode_b64": ([1] * 64, [8191 + 37 * s for s in range(64)], True),
    # PR sgl-project/sglang-torchtpu#2198 baseline (exp-6mtkc6t2qi), per attention-DP rank
    # (TP16 / attn-DP16): C16 extend 4096 on a 49,152 cache in 2048 chunks, one request per
    # rank; C64 decode on ~90K context, four requests per rank, NEXTN 6 draft tokens.
    "pr2198_extend_c1": ([2048], [49152], "extend"),
    "pr2198_extend_c2": ([2048], [51200], "extend"),
    "pr2198_verify_b4": ([6] * 4, [90000 + 37 * s for s in range(4)], "verify"),
    "pr2198_decode_b4": ([1] * 4, [90005 + 37 * s for s in range(4)], "decode"),
}
DEFAULT_CASES = ["extend_2048", "extend_4x2048", "extend_unaligned", "decode_b64"]


def _bf16(a):
    import ml_dtypes

    return np.asarray(a, np.float32).astype(ml_dtypes.bfloat16)


def case(name, seed=0):
    import ml_dtypes

    q_lens, prefixes, mode = CASES[name]
    mode = {False: "extend", True: "decode"}.get(mode, mode)
    decode = mode == "decode"
    w = np.random.default_rng(seed)
    r = np.random.default_rng(seed + 1 + sum(q_lens) + len(q_lens))
    S, T = len(q_lens), int(sum(q_lens))
    kv = np.add(prefixes, q_lens).astype(np.int32)
    ppr = int(-(-kv.max() // FP))
    # Request s owns framework pages in reverse order; page 0 is never real.
    pi = np.concatenate([np.arange(1 + (s + 1) * ppr - 1, s * ppr, -1) for s in range(S)])
    pos = np.concatenate([np.arange(p, p + q) for p, q in zip(prefixes, q_lens)])
    ang = r.uniform(0, 2 * np.pi, (T, 32))
    return dict(
        name=name, mode=mode, decode=decode, prefill=mode == "extend", S=S, T=T, ppr=ppr,
        kv=kv, pi=pi.astype(np.int32), pos=pos, prefixes=np.asarray(prefixes),
        cu_q=np.r_[0, np.cumsum(q_lens)].astype(np.int32),
        dist=np.array([S] * 3 if decode else [0, 0, S], np.int32),
        cache=r.normal(0, 1, (1 + S * ppr, FP, D)).astype(ml_dtypes.bfloat16),
        hidden=r.normal(0, 1, (T, HIDDEN)).astype(ml_dtypes.bfloat16),
        wk=w.normal(0, 0.02, (D, HIDDEN)).astype(ml_dtypes.float8_e4m3fn),  # [N, K]
        ws=w.uniform(0.5, 1.5, D).astype(np.float32),
        nw=w.uniform(0.5, 1.5, D).astype(ml_dtypes.bfloat16),
        nb=w.normal(0, 0.1, D).astype(ml_dtypes.bfloat16),
        query=r.normal(0, 1, (T, HEADS, D)).astype(ml_dtypes.bfloat16),
        cos=np.cos(ang).astype(ml_dtypes.bfloat16), sin=np.sin(ang).astype(ml_dtypes.bfloat16),
        w=r.uniform(0.1, 1.0, (T, HEADS)).astype(np.float32),
    )


def hadamard():
    h = np.ones((1, 1), np.float32)
    while h.shape[0] < D:
        h = np.block([[h, h], [h, -h]])
    return (h * D**-0.5).astype(np.float32)


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


def new_slots(c):
    """(framework page, offset) of every new token."""
    out = []
    for s in range(c["S"]):
        q = c["cu_q"][s + 1] - c["cu_q"][s]
        p = c["prefixes"][s] + np.arange(q)
        out.append(np.stack([c["pi"][s * c["ppr"] + p // FP], p % FP], 1))
    return np.concatenate(out)


def shortfall(c, rows, keys_by_seq, q, queries=32):
    """1 - (lowest exact score picked) / (exact k-th best), f64, on sampled queries."""
    seq = np.searchsorted(c["cu_q"][1:], np.arange(c["T"]), side="right")
    rows = np.asarray(rows)
    vals = []
    for t in np.linspace(0, c["T"] - 1, min(queries, c["T"])).astype(int):
        s = seq[t]
        n = c["kv"][s] - (c["cu_q"][s + 1] - c["cu_q"][s]) + (t - c["cu_q"][s]) + 1
        kk = keys_by_seq(s)[:n]
        sc = (np.maximum(q[t] @ kk.T, 0) * c["w"][t][:, None]).sum(0)
        picked = rows[t][rows[t] >= 0]
        if n <= K:
            vals.append(0.0 if len(picked) == n else 1.0)
            continue
        kth = -np.partition(-sc, K - 1)[K - 1]
        vals.append(max(0.0, 1 - sc[picked].min() / kth) if len(picked) == K else 1.0)
    v = np.asarray(vals)
    return {"max": float(v.max()), "p99": float(np.quantile(v, 0.99)),
            "frac_over_2pct": float(np.mean(v > 0.02))}


def framework_keys(c, cache):
    """Sequence-local keys of request s from a framework-page cache [P, 128, 128]."""
    cache = np.asarray(cache).astype(np.float64)
    return lambda s: cache[c["pi"][s * c["ppr"]:(s + 1) * c["ppr"]]].reshape(-1, D)


def _to_torch(torch, x):
    """NumPy array (bf16 / fp8 via ml_dtypes, int32, f32) -> TPU tensor, same bytes."""
    x = np.ascontiguousarray(x)
    if str(x.dtype) == "bfloat16":
        return torch.from_numpy(x.view(np.int16)).view(torch.bfloat16).to("tpu")
    if str(x.dtype) == "float8_e4m3fn":
        return torch.from_numpy(x.view(np.uint8)).view(torch.float8_e4m3fn).to("tpu")
    return torch.from_numpy(x).to("tpu")


def stage_torch(a):
    os.environ["JAX_PLATFORMS"] = "cpu"  # TorchTPU owns the device
    import torch
    import torch_tpu  # noqa: F401

    from bench_torch_ab import _profile
    from sglang_tpu.dsa_bf16_indexer_op import build_dsa_bf16_indexer_op

    op = build_dsa_bf16_indexer_op()
    sync = torch.accelerator.synchronize
    H = torch.from_numpy(hadamard()).to("tpu")
    rows = {}
    for name in a.cases:
        c = case(name)
        t = lambda x: _to_torch(torch, x)
        inputs = (t(c["hidden"]), t(c["wk"]), t(c["ws"]), t(c["nw"]), t(c["nb"]), t(c["query"]),
                  t(c["cos"]), t(c["sin"]), t(c["w"]), t(c["kv"]), t(c["pi"]), t(c["cu_q"]),
                  t(c["dist"]), H)
        static = (K, c["prefill"], c["decode"])

        def run(cache, *x):
            return op(cache, *x, *static)

        fn = torch.compile(run, backend="tpu", fullgraph=True, dynamic=False)
        cache = t(c["cache"])
        out = fn(cache, *inputs).cpu().numpy()
        sync()
        np.savez(os.path.join(a.npz_dir, f"chain_torch_{name}.npz"), rows=out,
                 cache=cache.cpu().view(torch.int16).numpy())
        row = {"case": name}
        bench_cache = t(c["cache"])
        row["us"], row["runs"] = _median_time(lambda: fn(bench_cache, *inputs), sync, a.iters, a.repeats)
        if a.xprof_dir:
            row["xprof"] = _profile(fn, (bench_cache, *inputs), a.iters, f"{a.xprof_dir}/chain_torch_{name}")
        rows[name] = row
        print("CTORCH " + json.dumps({k: v for k, v in row.items() if k != "xprof"}), flush=True)
    with open(os.path.join(a.npz_dir, "chain_torch_rows.json"), "w") as f:
        json.dump(rows, f, indent=1)


def stage_jax(a):
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    from xplane_summary import summarize
    from sglang_tpu.attention.pallas import dsa_bf16_indexer as ix
    from sglang_tpu.attention.pallas import dsa_index_topk as dit
    from sglang_tpu.attention.pallas.dsa_bf16_query import expanded_coefficients, query_tail_xla
    from sglang_tpu.attention.pallas.dsa_key_prologue import key_chain_xla
    from sglang_tpu.attention.pallas.util import quantize_fp8_ue8m0

    for p in (os.path.dirname(a.source_k), os.path.dirname(a.source_q), a.source_root):
        sys.path.insert(0, p)
    src_k = __import__(os.path.basename(a.source_k)[:-3]).fused_fp8_k_projection_scatter_pallas
    src_q = __import__(os.path.basename(a.source_q)[:-3]).fused_q_rope_hadamard
    from sgl_jax.srt.kernels.dsa import indexer as src_idx  # noqa: E402

    with open(os.path.join(a.npz_dir, "chain_torch_rows.json")) as f:
        torch_rows = json.load(f)
    H = jnp.asarray(hadamard())

    def timed(f, args, donate, tag):
        """Run, time and profile ``f(*args)``; arg ``donate`` is the cache, threaded through."""
        jf = jax.jit(f, donate_argnums=(donate,))
        state = {"a": list(args)}

        def call():
            topk, cache = jf(*state["a"])
            state["a"][donate] = cache
            state["topk"] = topk

        sync = lambda: jax.block_until_ready(state["a"][donate])
        med, runs = _median_time(call, sync, a.iters, a.repeats)
        res = {"us": med, "runs": runs}
        if a.xprof_dir:
            d = f"{a.xprof_dir}/chain_jax_{tag}"
            sync()
            jax.profiler.start_trace(d)
            for _ in range(a.iters):
                call()
            sync()
            jax.profiler.stop_trace()
            try:
                res["xprof"] = summarize(d, a.iters)
            except Exception as e:  # report, do not hide
                res["xprof"] = {"error": repr(e)[:2000]}
        return res

    report = {"jax": jax.__version__, "device": str(jax.devices()[0]), "rows": []}
    for name in a.cases:
        c = case(name)
        j = {k: jnp.asarray(c[k]) for k in ("hidden", "wk", "ws", "nw", "nb", "query", "cos",
                                             "sin", "w", "kv", "pi", "cu_q", "dist")}
        row = {"case": name, "T": c["T"], "decode": c["decode"]}
        slots = new_slots(c)
        keep = np.ones(c["cache"].shape[0], bool)
        keep[np.unique(slots[:, 0])] = False
        before16 = c["cache"].view(np.int16)

        # Ported-S from JAX.
        def ported(cache, *x):
            return ix.bf16_indexer_step(cache, *x, H, topk=K, prefill=c["prefill"], decode=c["decode"])

        p_args = (j["hidden"], j["wk"], j["ws"], j["nw"], j["nb"], j["query"], j["cos"], j["sin"],
                  j["w"], j["kv"], j["pi"], j["cu_q"], j["dist"])
        p_rows, p_cache = jax.jit(ported)(jnp.asarray(c["cache"]), *p_args)
        p_rows, p_cache = np.asarray(p_rows), np.asarray(p_cache)
        q_ported = np.asarray(ix.index_query_tail(j["query"], j["cos"], j["sin"], H, prefill=c["prefill"]))
        row["ported_s_jax"] = {
            "untouched": bool(np.array_equal(p_cache.view(np.int16)[keep], before16[keep])),
            "selection": shortfall(c, p_rows, framework_keys(c, p_cache), q_ported.astype(np.float64)),
            **timed(lambda cache, *x: ported(cache, *x), (jnp.asarray(c["cache"]), *p_args), 0,
                    f"{name}_ported_s_jax"),
        }

        # Source, chained by hand at page 64.
        table64 = np.stack([2 * c["pi"], 2 * c["pi"] + 1], 1).reshape(-1).astype(np.int32)
        cukv = (np.arange(c["S"] + 1) * c["ppr"] * FP).astype(np.int32)
        if c["decode"]:
            tp = np.array([table64[s * 2 * c["ppr"] + (c["kv"][s] - 1) // 64] * 64 + (c["kv"][s] - 1) % 64
                           for s in range(c["S"])], np.int32)
        else:
            ok = c["T"] % 64 == 0 and all(p % 64 == 0 for p in c["prefixes"])
            tp = None if not ok else np.concatenate([
                table64[s * 2 * c["ppr"] + (c["prefixes"][s] + np.arange(0, q, 64)) // 64]
                for s, q in enumerate(np.diff(c["cu_q"]))]).astype(np.int32)
        if tp is None:
            row["source"] = {"note": "unaligned extend: the source runs its XLA chain (dsa_indexer_ops, "
                                     "not shipped); Ported-S is compared against page 64 in the unit test"}
        else:
            tp_j, t64, cukv_j = jnp.asarray(tp), jnp.asarray(table64), jnp.asarray(cukv)

            def source(cache, hidden, wk, ws, nw, nb, query, cos, sin, w, kv, pi, cu_q, dist):
                view = cache.reshape(-1, 64, D)
                view = src_k(hidden, wk, ws, nw, nb, cos, sin, H, view, tp_j, page_size=64,
                             token_slots=c["decode"])
                if c["prefill"]:
                    ec, es = expanded_coefficients(cos, sin)
                    q = src_q(query, ec, es, H, token_tile=64)
                else:
                    q = query_tail_xla(query, cos, sin, H)
                topk = src_idx.compute_scores_and_select_topk_indices(
                    q, w, view, kv, t64, cu_q, cukv_j, dist, k=K, pages_per_seq=2 * c["ppr"],
                    one_token_per_seq=c["decode"], topk_impl="radix", output_physical_slots=False)
                return topk, view.reshape(cache.shape)

            s_rows, s_cache = jax.jit(source)(jnp.asarray(c["cache"]), *p_args)
            s_rows, s_cache = np.asarray(s_rows), np.asarray(s_cache)
            row["source"] = {
                "untouched": bool(np.array_equal(s_cache.view(np.int16)[keep], before16[keep])),
                "selection": shortfall(c, s_rows, framework_keys(c, s_cache), q_ported.astype(np.float64)),
                **timed(source, (jnp.asarray(c["cache"]), *p_args), 0, f"{name}_source"),
            }
            row["ported_s_jax_vs_source_cache"] = bool(np.array_equal(p_cache.view(np.int16), s_cache.view(np.int16)))
            row["ported_s_jax_vs_source_topk"] = bool(np.array_equal(np.sort(p_rows, 1), np.sort(s_rows, 1)))

        # Ported-S through the torch op.
        t = np.load(os.path.join(a.npz_dir, f"chain_torch_{name}.npz"))
        row["ported_s"] = {
            "untouched": bool(np.array_equal(t["cache"][keep], before16[keep])),
            **{k: v for k, v in torch_rows[name].items()},
        }
        row["ported_s_vs_ported_s_jax_cache"] = bool(np.array_equal(t["cache"], p_cache.view(np.int16)))
        row["ported_s_vs_ported_s_jax_topk"] = bool(np.array_equal(np.sort(t["rows"], 1), np.sort(p_rows, 1)))

        # Target: FP8 indexer step on its own key chain (W8A8, eps 1e-6, no Hadamard), page 128.
        try:
            # Same prefix values as the bf16 cache, stored as Target records by the
            # target's own writer (every slot of every page, page table 0..P-1).
            num_pages = c["cache"].shape[0]
            shape = dit.index_key_cache_shape(num_pages, FP, D)
            kf, _, kb = quantize_fp8_ue8m0(jnp.asarray(c["cache"]).reshape(-1, D))
            t_cache0 = dit._store_keys(
                jnp.zeros(shape, jnp.uint8), kf, kb, jnp.array([num_pages * FP], jnp.int32),
                jnp.arange(num_pages, dtype=jnp.int32), jnp.array([0, num_pages * FP], jnp.int32),
                jnp.array([0, 0, 1], jnp.int32))
            wk_kn = j["wk"].T
            q_rope = jnp.asarray(np.asarray(query_tail_xla(j["query"], j["cos"], j["sin"], jnp.eye(D)),
                                            np.float32).astype(c["query"].dtype))

            def target(cache, hidden, query, w, kv, pi, cu_q, dist):
                key = key_chain_xla(hidden, wk_kn, j["ws"], j["nw"].astype(jnp.float32),
                                    j["nb"].astype(jnp.float32), j["cos"], j["sin"], eps=1e-6)
                topk, cache = dit.dsa_indexer_step(query, key, w, cache, kv, pi, cu_q, dist,
                                                   weight_scale=1.0, topk=K)
                return topk, cache

            t_args = (t_cache0, j["hidden"], q_rope, j["w"], j["kv"], j["pi"], j["cu_q"], j["dist"])
            t_rows, t_cache = jax.jit(target)(*t_args)
            t_rows = np.asarray(t_rows)
            # Record of slot o of page p: [p, o // 4, o % 4] -> 128 FP8 bytes, then the UE8M0 byte.
            rec = np.asarray(t_cache).reshape(-1, shape[-1])
            deq = (rec[:, :D].view(c["wk"].dtype).astype(np.float64)
                   * np.exp2(rec[:, D:D + 1].astype(np.float64) - 127)).reshape(-1, FP, D)
            row["target"] = {
                "note": "FP8 UE8M0 records, page 128, W8A8 eps 1e-6, no Hadamard",
                "selection_own_fp8": shortfall(c, t_rows, framework_keys(c, deq),
                                               np.asarray(q_rope, np.float64)),
                **timed(target, t_args, 0, f"{name}_target"),
            }
        except Exception as e:  # report, do not hide
            row["target"] = {"error": repr(e)[:3000]}
        print("CHAIN " + json.dumps(_summary(row)), flush=True)
        report["rows"].append(row)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)


def _summary(row):
    out = {k: v for k, v in row.items() if not isinstance(v, dict)}
    for side in ("source", "ported_s_jax", "ported_s", "target"):
        r = row.get(side, {})
        x = r.get("xprof", {})
        out[side] = {
            "wall_us": r.get("us"),
            "module_us": round(sum(u for _, u in x.get("modules", [])), 1) if "modules" in x else None,
            "top_ops": [(n[:50], round(u, 1)) for n, u in x.get("ops", [])[:5]],
            **{k: r[k] for k in ("untouched", "selection", "selection_own_fp8", "note", "error") if k in r},
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("torch", "jax"), required=True)
    p.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--npz-dir", default="/tmp/chain")
    p.add_argument("--xprof-dir", default="")
    p.add_argument("--source-k", default="/tmp/source/fused_fp8_k_projection_scatter.py")
    p.add_argument("--source-q", default="/tmp/source/fused_q_rope_hadamard.py")
    p.add_argument("--source-root", default="/tmp/source/m3s")
    p.add_argument("--out", default="/tmp/chain/report.json")
    a = p.parse_args()
    os.makedirs(a.npz_dir, exist_ok=True)
    stage_torch(a) if a.stage == "torch" else stage_jax(a)


if __name__ == "__main__":
    main()
