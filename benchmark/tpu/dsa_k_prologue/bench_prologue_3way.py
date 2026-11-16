"""Standalone index-key prologue: Source / Target / Ported, correctness and speed.

"hidden -> index keys stored in the cache", no top-k. Each side under its own
contract (prologue_common.py):

* Source: sglang-jax-private ``fused_fp8_k_projection_scatter_pallas`` at a
  pinned commit, passed in with ``--source-kernel`` (never committed here).
  JAX, bf16 cache, page 64, Hadamard, W8A16.
* Target: the torch_tpu chain ``indexer_forward`` runs today (``apply_tpu_fp8_linear``,
  f32 LayerNorm, the TPU rope path) + ``quantize_fp8_ue8m0`` + ``_store_keys``,
  the store wrapped as a torch op of its own.
* Ported: the rope-table gather + ``store_keys_fused`` as a torch op (the
  fused kernel on these page-aligned batches).
* ported_jax: ``store_keys_fused`` called from JAX, to compare with Source in
  the same framework.

Stages: ``--stage torch`` (Target, Ported; saves their records) then
``--stage jax`` (Source, ported_jax; scores everything). Speed is the median of
``--repeats`` runs of ``--iters`` chained calls, plus xprof device time per call.
The prefix cache holds valid records (prefix_cache.py), not random bytes.
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
import prologue_common as pc  # noqa: E402

PAGE = 128
# References are f64 over [tokens, 6144]; score at most this many tokens.
REF_TOKENS = 8192


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


def _new_records(cache, slots):
    """Records of the new tokens from a ``u8[P, page // 4, 4, 256]`` cache."""
    p, s = slots[:, 0], slots[:, 1]
    return cache[p, s // 4, s % 4]


def _untouched(after, before, slots):
    keep = np.ones(after.shape[0], bool)
    keep[np.unique(slots[:, 0])] = False
    return bool(np.array_equal(after[keep], before[keep]))


def stage_torch(a):
    os.environ["JAX_PLATFORMS"] = "cpu"  # TorchTPU owns the device
    import jax
    import torch
    import torch.nn.functional as F
    import torch_tpu  # noqa: F401
    from torch_tpu._internal import pallas as tp

    from bench_torch_ab import _profile
    from prefix_cache import valid_prefix_cache
    from sglang_tpu.attention.pallas import dsa_index_topk as dit
    from sglang_tpu.attention.pallas.util import quantize_fp8_ue8m0
    from sglang_tpu.fp8_linear import apply_tpu_fp8_linear
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    dev, D, ROPE = "tpu", pc.D, pc.ROPE

    def store(key: jax.Array, cache: jax.Array, kv_lens: jax.Array, pages: jax.Array,
              cu: jax.Array, dist: jax.Array) -> jax.Array:
        kf, _, kb = quantize_fp8_ue8m0(key)
        return dit._store_keys(cache, kf, kb, kv_lens, pages, cu, dist)

    def fused(cache: jax.Array, hidden: jax.Array, wk: jax.Array, ws: jax.Array,
              nw: jax.Array, nb: jax.Array, cos: jax.Array, sin: jax.Array,
              kv_lens: jax.Array, pages: jax.Array, cu: jax.Array, dist: jax.Array,
              eps: float) -> jax.Array:
        return dit.store_keys_fused(cache, hidden, wk, ws, nw, nb, cos, sin,
                                    kv_lens, pages, cu, dist, eps=eps)

    store_k = tp.custom_jax_kernel(store, name="jax_m1bench_store", donate_argnums=(1,),
                                   input_output_aliases={1: 0}, warn_deprecated=False)
    fused_k = tp.custom_jax_kernel(fused, name="jax_m1bench_fused", static_argnums=(12,),
                                   donate_argnums=(0,), input_output_aliases={0: 0},
                                   warn_deprecated=False)

    @torch.library.custom_op("m1bench::store_keys", mutates_args=("cache",), device_types=["tpu"])
    def store_op(key: torch.Tensor, cache: torch.Tensor, kv_lens: torch.Tensor,
                 pages: torch.Tensor, cu: torch.Tensor, dist: torch.Tensor) -> None:
        with tpu_info_for_cpu_export():
            store_k(key, cache, kv_lens, pages, cu, dist)

    store_op.register_fake(lambda *args: None)

    @torch.library.custom_op("m1bench::store_keys_fused", mutates_args=("cache",), device_types=["tpu"])
    def fused_op(cache: torch.Tensor, hidden: torch.Tensor, wk: torch.Tensor, ws: torch.Tensor,
                 nw: torch.Tensor, nb: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                 kv_lens: torch.Tensor, pages: torch.Tensor, cu: torch.Tensor,
                 dist: torch.Tensor, eps: float) -> None:
        with tpu_info_for_cpu_export():
            fused_k(cache, hidden, wk, ws, nw, nb, cos, sin, kv_lens, pages, cu, dist, eps)

    fused_op.register_fake(lambda *args: None)

    bf = lambda arr: torch.from_numpy(np.ascontiguousarray(arr).view(np.int16)).view(torch.bfloat16)
    i0 = pc.inputs(a.tokens[0])
    wk = torch.from_numpy(i0["wk"].view(np.uint8)).view(torch.float8_e4m3fn).to(dev)
    ws, nw, nb = (torch.from_numpy(i0[k]).to(dev) for k in ("ws", "nw", "nb"))
    table = bf(i0["table"])  # the rope's bf16 cos_sin_cache, [positions, 64]
    cos, sin = table.chunk(2, -1)
    tpu_table = torch.cat([cos.repeat_interleave(2, -1), sin.repeat_interleave(2, -1)], -1).to(dev)
    rot = torch.zeros(ROPE, ROPE, dtype=torch.bfloat16)
    idx = torch.arange(ROPE)
    rot[idx[1::2], idx[0::2]] = -1.0
    rot[idx[0::2], idx[1::2]] = 1.0
    rot, table = rot.to(dev), table.to(dev)

    def target(hidden, positions, cache, kv, pages, cu, dist):
        k = apply_tpu_fp8_linear(hidden, wk, ws)
        k = F.layer_norm(k.float(), (D,), nw, nb, pc.EPS_TARGET).to(torch.bfloat16)
        cs = tpu_table[positions]
        kr = k[:, :ROPE]
        kr = kr * cs[:, :ROPE] + (kr @ rot) * cs[:, ROPE:]
        k = torch.cat([kr.to(k.dtype), k[:, ROPE:]], -1)
        store_op(k.contiguous(), cache, kv, pages, cu, dist)

    def ported(hidden, positions, cache, kv, pages, cu, dist):
        c, s = table.index_select(0, positions).chunk(2, -1)
        fused_op(cache, hidden, wk, ws, nw, nb, c.contiguous(), s.contiguous(),
                 kv, pages, cu, dist, pc.EPS_TARGET)

    compile_ = lambda f: torch.compile(f, backend="tpu", fullgraph=True, dynamic=False)
    variants = {"target": compile_(target), "ported": compile_(ported)}
    sync = torch.accelerator.synchronize
    rows = []
    for m in a.tokens:
        i, L = pc.inputs(m), pc.layout(m, PAGE)
        assert np.array_equal(i["wk"].view(np.uint8), i0["wk"].view(np.uint8))
        prefix_np = valid_prefix_cache(L["num_pages"], seed=m)
        prefix = torch.from_numpy(prefix_np)
        hidden = bf(i["x"]).to(dev)
        positions = torch.from_numpy(i["pos"].astype(np.int64)).to(dev)
        meta = [torch.from_numpy(L[k]).to(dev) for k in ("kv_lens", "page_indices", "cu_q_lens", "distribution")]
        row, recs = {"tokens": m, "requests": L["seqs"]}, {}
        for name, fn in variants.items():
            cache = prefix.clone().to(dev)
            fn(hidden, positions, cache, *meta)
            sync()
            after = cache.cpu().numpy()
            recs[name] = _new_records(after, L["slots"])
            row[f"{name}_prefix_untouched"] = _untouched(after, prefix_np, L["slots"])
            args = (hidden, positions, prefix.clone().to(dev), *meta)
            row[f"{name}_us"], row[f"{name}_runs"] = _median_time(
                lambda: fn(*args), sync, a.iters, a.repeats)
            if a.xprof_dir:
                row[f"{name}_xprof"] = _profile(fn, args, a.iters, f"{a.xprof_dir}/torch_{m}_{name}")
        np.savez(os.path.join(a.npz_dir, f"torch_{m}.npz"), **recs)
        print("TORCH " + json.dumps(row), flush=True)
        rows.append(row)
    with open(os.path.join(a.npz_dir, "torch_rows.json"), "w") as f:
        json.dump(rows, f, indent=1)


def stage_jax(a):
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    from xplane_summary import summarize
    from sglang_tpu.attention.pallas import dsa_index_topk as dit

    sys.path.insert(0, os.path.dirname(a.source_kernel))
    source = __import__(os.path.basename(a.source_kernel)[:-3]).fused_fp8_k_projection_scatter_pallas
    with open(os.path.join(a.npz_dir, "torch_rows.json")) as f:
        torch_rows = {r["tokens"]: r for r in json.load(f)}
    H = pc.hadamard()

    def run(fn, args, cache, tag):
        f = jax.jit(fn, donate_argnums=(len(args),))
        state = {"c": f(*args, cache)}
        call = lambda: state.__setitem__("c", f(*args, state["c"]))
        sync = lambda: jax.block_until_ready(state["c"])
        med, runs = _median_time(call, sync, a.iters, a.repeats)
        out = {"us": med, "runs": runs}
        if a.xprof_dir:
            d = f"{a.xprof_dir}/jax_{tag}"
            sync()
            jax.profiler.start_trace(d)
            for _ in range(a.iters):
                call()
            sync()
            jax.profiler.stop_trace()
            try:
                out["xprof"] = summarize(d, a.iters)
            except Exception as e:  # report, do not hide
                out["xprof"] = {"error": repr(e)[:2000]}
        return out

    report = {"source_kernel_sha256": a.source_sha256, "jax": jax.__version__,
              "device": str(jax.devices()[0]), "rows": []}
    for m in a.tokens:
        i = pc.inputs(m)
        n = min(m, REF_TOKENS)
        S = pc.SELECT_TOKENS
        row = {"tokens": m, "ref_tokens": n, "select_tokens": S}
        cos = jnp.asarray(i["table"][i["pos"], : pc.ROPE // 2])
        sin = jnp.asarray(i["table"][i["pos"], pc.ROPE // 2 :])
        x = jnp.asarray(i["x"])

        # Source: its own contract and cache.
        L64 = pc.layout(m, 64)
        tp64 = jnp.asarray(L64["slots"][::64, 0].astype(np.int32))
        wk_nk = jnp.asarray(i["wk"].T.copy())
        src_args = (x, wk_nk, jnp.asarray(i["ws"]), jnp.asarray(pc.bf16(i["nw"]).astype(pc.BF16)),
                    jnp.asarray(pc.bf16(i["nb"]).astype(pc.BF16)), cos, sin, jnp.asarray(H))
        src_fn = lambda *args: source(*args[:-1], args[-1], tp64, page_size=64)
        prefix64 = pc.prefix_bf16(L64["num_pages"], seed=m)
        try:
            after = np.asarray(src_fn(*src_args, jnp.asarray(prefix64))).astype(np.float32)
            sl = L64["slots"]
            keys = after[sl[:, 0], sl[:, 1]] @ H  # undo the orthonormal, symmetric Hadamard
            ref = pc.source_reference({**i, "x": i["x"][:n], "pos": i["pos"][:n]})
            err = np.abs(keys[:n] - ref)
            keep = np.ones(after.shape[0], bool)
            keep[np.unique(sl[:, 0])] = False
            row["source"] = {
                "prefix_untouched": bool(np.array_equal(after[keep], prefix64[keep].astype(np.float32))),
                "max_abs_vs_own_ref": float(err.max()),
                "mean_abs_vs_own_ref": float(err.mean()),
                "rel_max_vs_own_ref": float(err.max() / np.abs(ref).max()),
                "selection_vs_own_ref": pc.shortfall(keys[:S], ref[:S]),
                **run(src_fn, src_args, jnp.asarray(prefix64), f"{m}_source"),
            }
        except Exception as e:  # report, do not hide
            row["source"] = {"error": repr(e)[:3000]}

        # Ported from JAX: same kernel as the torch op, same framework as Source.
        L = pc.layout(m, PAGE)
        from prefix_cache import valid_prefix_cache

        prefix = valid_prefix_cache(L["num_pages"], seed=m)
        meta = tuple(jnp.asarray(L[k]) for k in ("kv_lens", "page_indices", "cu_q_lens", "distribution"))
        w = (jnp.asarray(i["wk"]), jnp.asarray(i["ws"]), jnp.asarray(i["nw"]), jnp.asarray(i["nb"]))
        pj_fn = lambda x, c, s, cache: dit.store_keys_fused(cache, x, *w, c, s, *meta, eps=pc.EPS_TARGET)
        pj_rec = _new_records(np.asarray(pj_fn(x, cos, sin, jnp.asarray(prefix))), L["slots"])
        row["ported_jax"] = run(pj_fn, (x, cos, sin), jnp.asarray(prefix), f"{m}_ported_jax")

        # Target and Ported (torch): records against the f64 reference of their contract.
        t = np.load(os.path.join(a.npz_dir, f"torch_{m}.npz"))
        ref_rec = pc.records(pc.target_reference({**i, "x": i["x"][:n], "pos": i["pos"][:n]}))
        exact = pc.dequant(ref_rec[:S])
        for name in ("target", "ported"):
            rec = t[name][:n]
            tr = torch_rows[m]
            row[name] = {
                "prefix_untouched": tr[f"{name}_prefix_untouched"],
                "exact_records_vs_ref": float(np.mean(np.all(rec == ref_rec, axis=1))),
                "scale_byte_max_diff": int(np.abs(rec[:, pc.D].astype(int) - ref_rec[:, pc.D]).max()),
                "selection_vs_own_ref": pc.shortfall(pc.dequant(rec[:S]), exact),
                "us": tr[f"{name}_us"], "runs": tr[f"{name}_runs"],
                **({"xprof": tr[f"{name}_xprof"]} if f"{name}_xprof" in tr else {}),
            }
        row["target_vs_ported_exact_records"] = float(np.mean(np.all(t["target"] == t["ported"], axis=1)))
        row["ported_torch_vs_jax_exact_records"] = float(np.mean(np.all(t["ported"] == pj_rec, axis=1)))
        print("ROW " + json.dumps(row), flush=True)
        print("SUM " + json.dumps(_summary(row)), flush=True)
        report["rows"].append(row)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)


def _summary(row):
    """The few numbers per point that fit in a log tail; the full row is in the report."""
    out = {"tokens": row["tokens"]}
    for name in ("source", "target", "ported", "ported_jax"):
        r = row.get(name, {})
        if "error" in r:
            out[name] = {"error": r["error"][:300]}
            continue
        x = r.get("xprof", {})
        out[name] = {
            "wall_us": r.get("us"),
            "device_us": x.get("device_us", x.get("error", "")[:200] or None),
            "prefix_untouched": r.get("prefix_untouched"),
            "sel_max": r.get("selection_vs_own_ref", {}).get("max"),
            "sel_p99": r.get("selection_vs_own_ref", {}).get("p99"),
            "sel_frac2": r.get("selection_vs_own_ref", {}).get("frac_over_2pct"),
            "exact_rec": r.get("exact_records_vs_ref"),
            "rel_max": r.get("rel_max_vs_own_ref"),
        }
    out["target_vs_ported_exact"] = row.get("target_vs_ported_exact_records")
    out["ported_torch_vs_jax_exact"] = row.get("ported_torch_vs_jax_exact_records")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("torch", "jax"), required=True)
    p.add_argument("--tokens", type=int, nargs="+", default=[2048, 8192, 32768])
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--npz-dir", default="/tmp/prologue3")
    p.add_argument("--xprof-dir", default="")
    p.add_argument("--source-kernel", default="")
    p.add_argument("--source-sha256", default="")
    p.add_argument("--out", default="/tmp/prologue3/report.json")
    a = p.parse_args()
    os.makedirs(a.npz_dir, exist_ok=True)
    stage_torch(a) if a.stage == "torch" else stage_jax(a)


if __name__ == "__main__":
    main()
