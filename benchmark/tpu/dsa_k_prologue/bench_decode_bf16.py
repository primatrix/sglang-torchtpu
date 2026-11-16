"""Decode index-key store, source contract (BF16 cache, page 64): Source / Ported-S.

One new token per request (``one_token_per_seq``): the source runs
``fused_fp8_k_projection_scatter_pallas(token_slots=True)``, a read-modify-write
of one cache row per token. Compared:

* Source: the pinned source kernel, ``--source-kernel`` (never committed here).
* ported_s_jax: ``store_index_keys_bf16(one_token_per_seq=True)`` called from JAX.
* ported_s: the ``dsa_store_index_keys_bf16`` torch op, compiled.

Same inputs, same cache, same computation for all three, so the new rows must be
byte-identical; each is also scored against the f64 source reference. Target has
no decode counterpart here (its decode store is part of the FP8 top-k op).

Requests sit at prefix ``PREFIX + s``, so neighbours land in different slots and
some share a page. ``--stage torch`` then ``--stage jax``.
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

PAGE = 64
PREFIX = 8192  # decode context; the store's cost does not depend on it


def batch(b, seed=0):
    """Inputs and metadata for ``b`` decode requests."""
    w = pc.inputs(pc.CHUNK, seed)  # weights only; identical for every token count
    r = np.random.default_rng(seed + 100 + b)
    ang = r.uniform(0, 2 * np.pi, (b, pc.ROPE // 2))
    table = np.concatenate([np.cos(ang), np.sin(ang)], -1).astype(pc.BF16)
    pps = (PREFIX + b + 1 + PAGE - 1) // PAGE
    pos = PREFIX + np.arange(b)
    pages = (1 + np.arange(b * pps)).astype(np.int32)
    slots = np.stack([pages[np.arange(b) * pps + pos // PAGE], pos % PAGE], 1)
    return dict(
        x=r.normal(0, 1, (b, pc.HIDDEN)).astype(pc.BF16),
        wk=w["wk"], ws=w["ws"], nw=w["nw"], nb=w["nb"],
        table=table, pos=np.arange(b),
        seq_lens=(pos + 1).astype(np.int32), page_indices=pages,
        cu_q_lens=np.arange(b + 1, dtype=np.int32),
        cu_kv_lens=(np.arange(b) * pps * PAGE).astype(np.int32),
        num_pages=b * pps + 1, slots=slots,
    )


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


def _untouched(after, before, slots):
    written = np.zeros(after.shape[:2], bool)
    written[slots[:, 0], slots[:, 1]] = True
    return bool(np.array_equal(after[~written], before[~written]))


def stage_torch(a):
    os.environ["JAX_PLATFORMS"] = "cpu"  # TorchTPU owns the device
    import torch
    import torch_tpu  # noqa: F401

    from bench_torch_ab import _profile
    from sglang_tpu.dsa_bf16_key_store_op import build_dsa_store_index_keys_bf16_op

    op = build_dsa_store_index_keys_bf16_op()
    dev = "tpu"
    bf = lambda arr: torch.from_numpy(np.ascontiguousarray(arr).view(np.int16)).view(torch.bfloat16)
    H = torch.from_numpy(pc.hadamard()).to(dev)

    def ported_s(hidden, wk, ws, nw, nb, cos, sin, cache, seq, pages, cuq, cukv):
        op(cache, hidden, wk, ws, nw, nb, cos, sin, H, seq, pages, cuq, cukv, True)

    fn = torch.compile(ported_s, backend="tpu", fullgraph=True, dynamic=False)
    sync = torch.accelerator.synchronize
    rows = []
    for b in a.batches:
        i = batch(b)
        before = pc.prefix_bf16(i["num_pages"], seed=b, page=PAGE)
        ins = (
            bf(i["x"]).to(dev),
            torch.from_numpy(i["wk"].T.copy().view(np.uint8)).view(torch.float8_e4m3fn).to(dev),
            torch.from_numpy(i["ws"]).to(dev), bf(pc.bf16(i["nw"]).astype(pc.BF16)).to(dev),
            bf(pc.bf16(i["nb"]).astype(pc.BF16)).to(dev),
            bf(i["table"][:, : pc.ROPE // 2]).to(dev), bf(i["table"][:, pc.ROPE // 2 :]).to(dev),
        )
        meta = [torch.from_numpy(i[k]).to(dev) for k in ("seq_lens", "page_indices", "cu_q_lens", "cu_kv_lens")]
        cache = bf(before).to(dev)
        fn(*ins, cache, *meta)
        sync()
        after = cache.cpu().view(torch.int16).numpy()
        sl = i["slots"]
        row = {"batch": b, "prefix_untouched": _untouched(after, before.view(np.int16), sl)}
        args = (*ins, bf(before).to(dev), *meta)
        row["us"], row["runs"] = _median_time(lambda: fn(*args), sync, a.iters, a.repeats)
        if a.xprof_dir:
            row["xprof"] = _profile(fn, args, a.iters, f"{a.xprof_dir}/dec_torch_{b}")
        np.save(os.path.join(a.npz_dir, f"dec_torch_{b}.npy"), after[sl[:, 0], sl[:, 1]])
        print("TORCH " + json.dumps({k: v for k, v in row.items() if k != "xprof"}), flush=True)
        rows.append(row)
    with open(os.path.join(a.npz_dir, "dec_torch_rows.json"), "w") as f:
        json.dump(rows, f, indent=1)


def stage_jax(a):
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    from xplane_summary import summarize
    from sglang_tpu.attention.pallas import dsa_bf16_key_store as ks

    sys.path.insert(0, os.path.dirname(a.source_kernel))
    source = __import__(os.path.basename(a.source_kernel)[:-3]).fused_fp8_k_projection_scatter_pallas
    with open(os.path.join(a.npz_dir, "dec_torch_rows.json")) as f:
        torch_rows = {r["batch"]: r for r in json.load(f)}
    H = pc.hadamard()

    def run(fn, args, cache, tag):
        f = jax.jit(fn, donate_argnums=(len(args),))
        state = {"c": f(*args, cache)}
        call = lambda: state.__setitem__("c", f(*args, state["c"]))
        sync = lambda: jax.block_until_ready(state["c"])
        med, runs = _median_time(call, sync, a.iters, a.repeats)
        out = {"us": med, "runs": runs}
        if a.xprof_dir:
            d = f"{a.xprof_dir}/dec_jax_{tag}"
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

    report = {"source_kernel_sha256": a.source_sha256, "rows": []}
    for b in a.batches:
        i = batch(b)
        sl = i["slots"]
        before = pc.prefix_bf16(i["num_pages"], seed=b, page=PAGE)
        x = jnp.asarray(i["x"])
        w = (jnp.asarray(i["wk"].T.copy()), jnp.asarray(i["ws"]),
             jnp.asarray(pc.bf16(i["nw"]).astype(pc.BF16)), jnp.asarray(pc.bf16(i["nb"]).astype(pc.BF16)))
        cos = jnp.asarray(i["table"][:, : pc.ROPE // 2])
        sin = jnp.asarray(i["table"][:, pc.ROPE // 2 :])
        meta = tuple(jnp.asarray(i[k]) for k in ("seq_lens", "page_indices", "cu_q_lens", "cu_kv_lens"))
        ref = pc.source_reference({**i, "pos": i["pos"]})
        flat = jnp.asarray((sl[:, 0] * PAGE + sl[:, 1]).astype(np.int32))

        def score(new_i16):
            keys = new_i16.view(pc.BF16).astype(np.float32) @ H
            err = np.abs(keys - ref)
            return {"max_abs_vs_own_ref": float(err.max()),
                    "rel_max_vs_own_ref": float(err.max() / np.abs(ref).max())}

        row = {"batch": b, "prefix": PREFIX}
        fns = {
            "source": lambda x, c, s, cache: source(x, *w, c, s, jnp.asarray(H), cache, flat,
                                                     page_size=PAGE, token_slots=True),
            "ported_s_jax": lambda x, c, s, cache: ks.store_index_keys_bf16(
                cache, x, *w, c, s, jnp.asarray(H), *meta, one_token_per_seq=True),
        }
        new = {}
        for name, f in fns.items():
            try:
                after = np.asarray(f(x, cos, sin, jnp.asarray(before))).view(np.int16)
                new[name] = after[sl[:, 0], sl[:, 1]]
                row[name] = {"prefix_untouched": _untouched(after, before.view(np.int16), sl),
                             **score(new[name]),
                             **run(f, (x, cos, sin), jnp.asarray(before), f"{b}_{name}")}
            except Exception as e:  # report, do not hide
                row[name] = {"error": repr(e)[:3000]}
        t = np.load(os.path.join(a.npz_dir, f"dec_torch_{b}.npy"))
        tr = torch_rows[b]
        row["ported_s"] = {"prefix_untouched": tr["prefix_untouched"], **score(t),
                           "us": tr["us"], "runs": tr["runs"],
                           **({"xprof": tr["xprof"]} if "xprof" in tr else {})}
        if "source" in new:
            for name, rec in (("ported_s_jax", new.get("ported_s_jax")), ("ported_s", t)):
                if rec is not None:
                    row[f"{name}_vs_source_exact"] = float(np.mean(np.all(rec == new["source"], axis=1)))
        report["rows"].append(row)
        print("DEC " + json.dumps(_summary(row)), flush=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)


def _summary(row):
    out = {"batch": row["batch"]}
    for name in ("source", "ported_s_jax", "ported_s"):
        r = row.get(name, {})
        if "error" in r:
            out[name] = {"error": r["error"][:300]}
            continue
        mods = r.get("xprof", {}).get("modules", [])
        out[name] = {
            "wall_us": r.get("us"),
            "module_us": round(sum(u for _, u in mods), 1) if mods else None,
            "top_ops": [(n[:60], round(u, 1)) for n, u in r.get("xprof", {}).get("ops", [])[:4]],
            "untouched": r.get("prefix_untouched"),
            "rel_max": r.get("rel_max_vs_own_ref"),
        }
    for k in ("ported_s_jax_vs_source_exact", "ported_s_vs_source_exact"):
        out[k] = row.get(k)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("torch", "jax"), required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[8, 64, 256])
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--npz-dir", default="/tmp/prologue3")
    p.add_argument("--xprof-dir", default="")
    p.add_argument("--source-kernel", default="")
    p.add_argument("--source-sha256", default="")
    p.add_argument("--out", default="/tmp/prologue3/decode_report.json")
    a = p.parse_args()
    os.makedirs(a.npz_dir, exist_ok=True)
    stage_torch(a) if a.stage == "torch" else stage_jax(a)


if __name__ == "__main__":
    main()
