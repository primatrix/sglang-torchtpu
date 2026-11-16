"""Locate where the torch op and the JAX call of the BF16 indexer chain diverge.

The chain matched byte for byte on aligned extend and decode (Pallas kernels
only) and differed on unaligned extend (T=2000), where the key store and the
query tail take their XLA chains. Here each stage runs on its own, through its
own torch op and from JAX, on identical inputs at page 64 (framework page p =
source pages 2p, 2p+1, the adapter's mapping):

1. key store (``dsa_store_index_keys_bf16``) -> cache
2. query tail (``dsa_index_query_tail_bf16``) -> q
3. scorer + radix top-k (``dsa_index_score_topk_bf16``) on stage 1-2 outputs

JAX is run with the default matmul precision and with "highest", to test
whether XLA precision defaults explain a stage difference. The scorer is also
run from JAX on the torch stage outputs, so a scorer difference is separated
from an input difference. extend_2048 is the aligned control. The radix ops of
each full-chain profile are listed with their counts (the torch op showed an
extra ``radix_topk_v2`` call at extend_2048).
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_indexer_chain as bc  # noqa: E402

K = bc.K


def page64(c):
    pi64 = np.stack([2 * c["pi"], 2 * c["pi"] + 1], 1).reshape(-1).astype(np.int32)
    cukv = (np.arange(c["S"] + 1) * c["ppr"] * bc.FP).astype(np.int32)
    cache64 = c["cache"].reshape(-1, 64, bc.D)
    return pi64, cukv, cache64


def stage_torch(a):
    os.environ["JAX_PLATFORMS"] = "cpu"
    import torch
    import torch_tpu  # noqa: F401

    from bench_torch_ab import _profile
    from sglang_tpu.dsa_bf16_index_score_op import build_dsa_index_score_topk_bf16_op
    from sglang_tpu.dsa_bf16_indexer_op import build_dsa_bf16_indexer_op
    from sglang_tpu.dsa_bf16_key_store_op import build_dsa_store_index_keys_bf16_op
    from sglang_tpu.dsa_bf16_query_op import build_dsa_index_query_tail_bf16_op

    store_op, q_op = build_dsa_store_index_keys_bf16_op(), build_dsa_index_query_tail_bf16_op()
    score_op, chain_op = build_dsa_index_score_topk_bf16_op(), build_dsa_bf16_indexer_op()
    t = lambda x: bc._to_torch(torch, x)
    H = torch.from_numpy(bc.hadamard()).to("tpu")
    comp = lambda f: torch.compile(f, backend="tpu", fullgraph=True, dynamic=False)
    for name in a.cases:
        c = bc.case(name)
        pi64, cukv, cache64 = page64(c)
        dec, pre = c["decode"], c["prefill"]

        def store(cache, *x):
            store_op(cache, *x, dec)

        cache = t(cache64)
        comp(store)(cache, t(c["hidden"]), t(c["wk"]), t(c["ws"]), t(c["nw"]), t(c["nb"]),
                    t(c["cos"]), t(c["sin"]), H, t(c["kv"]), t(pi64), t(c["cu_q"]), t(cukv))
        q = comp(lambda *x: q_op(*x, pre))(t(c["query"]), t(c["cos"]), t(c["sin"]), H)
        rows = comp(lambda *x: score_op(*x, K, 2 * c["ppr"], dec, False))(
            q, t(c["w"]), cache, t(c["kv"]), t(pi64), t(c["cu_q"]), t(cukv), t(c["dist"]))
        out = dict(cache=cache.cpu().view(torch.int16).numpy(), q=q.cpu().numpy(),
                   rows=rows.cpu().numpy())

        # Full chain op, and its radix ops.
        inputs = (t(c["hidden"]), t(c["wk"]), t(c["ws"]), t(c["nw"]), t(c["nb"]), t(c["query"]),
                  t(c["cos"]), t(c["sin"]), t(c["w"]), t(c["kv"]), t(c["pi"]), t(c["cu_q"]),
                  t(c["dist"]), H)
        fn = comp(lambda cc, *x: chain_op(cc, *x, K, pre, dec))
        full_cache = t(c["cache"])
        out["full_rows"] = fn(full_cache, *inputs).cpu().numpy()
        out["full_cache"] = full_cache.cpu().view(torch.int16).numpy()
        np.savez(os.path.join(a.npz_dir, f"diag_torch_{name}.npz"), **out)
        prof = _profile(fn, (t(c["cache"]), *inputs), a.iters, f"{a.xprof_dir}/diag_torch_{name}")
        radix = [(n[:70], round(u, 1)) for n, u in prof.get("ops", []) if "radix" in n]
        print("DTORCH " + json.dumps({"case": name, "radix_ops": radix,
                                      "module_us": sum(u for _, u in prof.get("modules", []))}),
              flush=True)


def _diff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    same = bool(np.array_equal(a, b))
    if same or a.dtype.kind not in "f":
        return {"exact": same, "elems_diff": int(np.sum(a != b))}
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"exact": False, "elems_diff": int(np.sum(d > 0)), "max_abs": float(d.max()),
            "rel_max": float(d.max() / max(np.abs(b).max(), 1e-30))}


def _bf16_cache_diff(a16, b16):
    import ml_dtypes

    f = lambda x: x.view(ml_dtypes.bfloat16).astype(np.float32)
    return _diff(f(a16), f(b16))


def _rows_diff(a, b):
    sa, sb = np.sort(a, 1), np.sort(b, 1)
    return {"exact": bool(np.array_equal(sa, sb)), "rows_diff": int(np.sum(np.any(sa != sb, 1)))}


def stage_jax(a):
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    from xplane_summary import summarize
    from sglang_tpu.attention.pallas import dsa_bf16_indexer as ix
    from sglang_tpu.attention.pallas.dsa_bf16_key_store import store_index_keys_bf16
    from sglang_tpu.attention.pallas.dsa_bf16_query import index_query_tail
    from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.indexer import (
        compute_scores_and_select_topk_indices as score,
    )

    H = jnp.asarray(bc.hadamard())
    for name in a.cases:
        c = bc.case(name)
        pi64, cukv, cache64 = page64(c)
        tor = np.load(os.path.join(a.npz_dir, f"diag_torch_{name}.npz"))
        dec, pre = c["decode"], c["prefill"]
        j = {k: jnp.asarray(c[k]) for k in ("hidden", "wk", "ws", "nw", "nb", "query", "cos",
                                             "sin", "w", "kv", "cu_q", "dist")}
        sk = dict(k=K, pages_per_seq=2 * c["ppr"], one_token_per_seq=dec, topk_impl="radix",
                  output_physical_slots=False)
        row = {"case": name}
        from sglang_tpu.attention.pallas import dsa_bf16_key_store as ks
        from sglang_tpu.attention.pallas.dsa_bf16_query import query_tail_xla

        def store_fn(cc):
            return store_index_keys_bf16(
                cc, j["hidden"], j["wk"], j["ws"], j["nw"], j["nb"], j["cos"], j["sin"], H,
                j["kv"], jnp.asarray(pi64), j["cu_q"], jnp.asarray(cukv), one_token_per_seq=dec)

        def q_fn():
            return index_query_tail(j["query"], j["cos"], j["sin"], H, prefill=pre)

        def score_fn(qq, cc):
            return score(qq, j["w"], cc, j["kv"], jnp.asarray(pi64), j["cu_q"],
                         jnp.asarray(cukv), j["dist"], **sk)

        def record(st, q, rows):
            return {
                "store_cache_vs_torch": _bf16_cache_diff(np.asarray(st).view(np.int16), tor["cache"]),
                "query_vs_torch": _diff(np.asarray(q), tor["q"]),
                "rows_vs_torch": _rows_diff(np.asarray(rows), tor["rows"]),
            }

        # 1. Plain jit on the TPU backend (the earlier "default" run).
        st = jax.jit(store_fn)(jnp.asarray(cache64))
        q = jax.jit(q_fn)()
        row["default"] = record(st, q, jax.jit(score_fn)(q, st))

        # 2. The torch op's path in this process: jax.export for "tpu", then call.
        #    If this matches torch, the difference is export lowering vs direct jit.
        def exported(f, *x):
            return jax.export.export(jax.jit(f), platforms=["tpu"]).call(*x)

        st_e = exported(store_fn, jnp.asarray(cache64))
        q_e = exported(q_fn)
        row["exported"] = record(st_e, q_e, exported(score_fn, q_e, st_e))
        row["exported_vs_default"] = {
            "cache": _bf16_cache_diff(np.asarray(st_e).view(np.int16), np.asarray(st).view(np.int16)),
            "query": _diff(np.asarray(q_e), np.asarray(q))}

        # 3. "highest" on the XLA chains only. The earlier run also wrapped the source
        #    Pallas K kernel, where Mosaic rejected the changed FP8 dot ("Bad lhs type").
        with jax.default_matmul_precision("highest"):
            key_h = jax.jit(lambda: ks.project_keys_xla(
                j["hidden"], j["wk"], j["ws"], j["nw"], j["nb"], j["cos"], j["sin"], H))()
            q_h = jax.jit(lambda: query_tail_xla(j["query"], j["cos"], j["sin"], H))()
        key_d = jax.jit(lambda: ks.project_keys_xla(
            j["hidden"], j["wk"], j["ws"], j["nw"], j["nb"], j["cos"], j["sin"], H))()
        row["highest_xla_only"] = {
            "key_vs_default": _diff(np.asarray(key_h).astype(np.float32), np.asarray(key_d).astype(np.float32)),
            "query_vs_torch": _diff(np.asarray(q_h), tor["q"]),
        }
        # Scorer alone on the torch stage outputs.
        import ml_dtypes

        tc = jnp.asarray(tor["cache"].view(ml_dtypes.bfloat16))
        r2 = np.asarray(jax.jit(lambda qq, cc: score(qq, j["w"], cc, j["kv"], jnp.asarray(pi64),
                                                     j["cu_q"], jnp.asarray(cukv), j["dist"], **sk))(
            jnp.asarray(tor["q"]), tc))
        row["scorer_on_torch_inputs_vs_torch_rows"] = _rows_diff(r2, tor["rows"])
        # Full chain from JAX vs the full torch op, and vs the stage-wise torch run.
        args = tuple(j[k] for k in ("hidden", "wk", "ws", "nw", "nb", "query", "cos", "sin",
                                    "w", "kv")) + (jnp.asarray(c["pi"]), j["cu_q"], j["dist"])
        chain = jax.jit(lambda cc, *x: ix.bf16_indexer_step(cc, *x, H, topk=K, prefill=pre,
                                                            decode=dec))
        fr, fc = chain(jnp.asarray(c["cache"]), *args)
        row["full_jax_vs_full_torch"] = {
            "cache": _bf16_cache_diff(np.asarray(fc).view(np.int16), tor["full_cache"]),
            "rows": _rows_diff(np.asarray(fr), tor["full_rows"])}
        row["full_torch_vs_stagewise_torch"] = {
            "cache": _bf16_cache_diff(tor["full_cache"].reshape(-1, 64, bc.D).reshape(tor["cache"].shape),
                                      tor["cache"]),
            "rows": _rows_diff(tor["full_rows"], tor["rows"])}
        # Radix ops in the JAX full-chain profile.
        d = f"{a.xprof_dir}/diag_jax_{name}"
        state = {"c": jnp.asarray(c["cache"])}
        jax.block_until_ready(chain(state["c"], *args))
        jax.profiler.start_trace(d)
        for _ in range(a.iters):
            o = chain(state["c"], *args)
        jax.block_until_ready(o)
        jax.profiler.stop_trace()
        try:
            prof = summarize(d, a.iters)
            row["jax_radix_ops"] = [(n[:70], round(u, 1)) for n, u in prof.get("ops", []) if "radix" in n]
        except Exception as e:  # report, do not hide
            row["jax_radix_ops"] = repr(e)[:500]
        print("DIAG " + json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("torch", "jax"), required=True)
    p.add_argument("--cases", nargs="+", default=["extend_unaligned", "extend_2048"])
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--npz-dir", default="/tmp/diag")
    p.add_argument("--xprof-dir", default="/tmp/diag/xprof")
    a = p.parse_args()
    os.makedirs(a.npz_dir, exist_ok=True)
    stage_torch(a) if a.stage == "torch" else stage_jax(a)


if __name__ == "__main__":
    main()
