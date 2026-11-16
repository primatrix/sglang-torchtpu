"""Indexer query tail (rope + Hadamard), prefill: Source / Ported-S / Target.

* Source: sglang-jax-private ``fused_q_rope_hadamard`` (pinned, ``--source-kernel``,
  never committed here), called as the source's call site does (token_tile 64,
  expanded coefficients).
* ported_s_jax / ported_s: ``dsa_bf16_query.index_query_tail(prefill=True)``,
  from JAX and as the ``dsa_index_query_tail_bf16`` torch op (compiled).
* target: the torch chain ``indexer_forward`` runs today for the query (TPU rope
  path: table gather, +-1 rotation dot, cat). Different contract: no Hadamard,
  bf16 output; reported for time only, not compared bytewise.

Source and Ported-S get identical inputs, so their f32 outputs must be
byte-identical; both are also scored against an f64 reference.
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
import prologue_common as pc  # noqa: E402

H, D, ROPE = 32, 128, 64


def inputs(tokens, seed=0):
    r = np.random.default_rng(seed + tokens)
    q = r.normal(0, 1, (tokens, H, D)).astype(pc.BF16)
    ang = r.uniform(0, 2 * np.pi, (tokens, ROPE // 2))
    return q, np.cos(ang).astype(pc.BF16), np.sin(ang).astype(pc.BF16)


def reference(q, cos, sin):
    """f64 except the bf16 rope rounding the source keeps."""
    q = q.astype(np.float64)
    c, s = cos.astype(np.float64)[:, None], sin.astype(np.float64)[:, None]
    x1, x2 = q[..., 0:ROPE:2].copy(), q[..., 1:ROPE:2].copy()
    q[..., 0:ROPE:2] = pc.bf16(x1 * c - x2 * s)
    q[..., 1:ROPE:2] = pc.bf16(x2 * c + x1 * s)
    return q @ pc.hadamard().astype(np.float64)


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


def stage_torch(a):
    os.environ["JAX_PLATFORMS"] = "cpu"
    import torch
    import torch_tpu  # noqa: F401

    from bench_torch_ab import _profile
    from sglang_tpu.dsa_bf16_query_op import build_dsa_index_query_tail_bf16_op

    dev = "tpu"
    op = build_dsa_index_query_tail_bf16_op()
    bf = lambda x: torch.from_numpy(np.ascontiguousarray(x).view(np.int16)).view(torch.bfloat16)
    had = torch.from_numpy(pc.hadamard()).to(dev)
    rot = torch.zeros(ROPE, ROPE, dtype=torch.bfloat16)
    idx = torch.arange(ROPE)
    rot[idx[1::2], idx[0::2]] = -1.0
    rot[idx[0::2], idx[1::2]] = 1.0
    rot = rot.to(dev)

    def target(q, table):  # table [T, 128]: cos, sin repeat-interleaved (TPU rope path)
        qr = q[..., :ROPE]
        c, s = table[:, None, :ROPE], table[:, None, ROPE:]
        qr = qr * c + (qr @ rot) * s
        return torch.cat([qr.to(q.dtype), q[..., ROPE:]], -1)

    def ported_s(q, cos, sin):
        return op(q, cos, sin, had, True)

    compile_ = lambda f: torch.compile(f, backend="tpu", fullgraph=True, dynamic=False)
    tgt, ps = compile_(target), compile_(ported_s)
    sync = torch.accelerator.synchronize
    rows = []
    for m in a.tokens:
        q, cos, sin = inputs(m)
        tq, tc, ts = bf(q).to(dev), bf(cos).to(dev), bf(sin).to(dev)
        table = torch.cat([bf(cos).repeat_interleave(2, -1), bf(sin).repeat_interleave(2, -1)], -1).to(dev)
        out = ps(tq, tc, ts).cpu().numpy()
        np.save(os.path.join(a.npz_dir, f"q_torch_{m}.npy"), out)
        row = {"tokens": m}
        for name, fn, args in (("ported_s", ps, (tq, tc, ts)), ("target", tgt, (tq, table))):
            row[f"{name}_us"], row[f"{name}_runs"] = _median_time(lambda: fn(*args), sync, a.iters, a.repeats)
            if a.xprof_dir:
                row[f"{name}_xprof"] = _profile(fn, args, a.iters, f"{a.xprof_dir}/q_torch_{m}_{name}")
        print("QTORCH " + json.dumps({k: v for k, v in row.items() if not k.endswith("xprof")}), flush=True)
        rows.append(row)
    with open(os.path.join(a.npz_dir, "q_torch_rows.json"), "w") as f:
        json.dump(rows, f, indent=1)


def stage_jax(a):
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    from xplane_summary import summarize
    from sglang_tpu.attention.pallas import dsa_bf16_query as dq

    sys.path.insert(0, os.path.dirname(a.source_kernel))
    source = __import__(os.path.basename(a.source_kernel)[:-3]).fused_q_rope_hadamard
    with open(os.path.join(a.npz_dir, "q_torch_rows.json")) as f:
        torch_rows = {r["tokens"]: r for r in json.load(f)}
    Hm = jnp.asarray(pc.hadamard())

    def run(fn, args, tag):
        f = jax.jit(fn)
        out = {"o": f(*args)}
        call = lambda: out.__setitem__("o", f(*args))
        sync = lambda: jax.block_until_ready(out["o"])
        med, runs = _median_time(call, sync, a.iters, a.repeats)
        res = {"us": med, "runs": runs}
        if a.xprof_dir:
            d = f"{a.xprof_dir}/q_jax_{tag}"
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

    report = {"source_kernel_sha256": a.source_sha256, "rows": []}
    for m in a.tokens:
        q, cos, sin = inputs(m)
        jq, jc, js = jnp.asarray(q), jnp.asarray(cos), jnp.asarray(sin)
        n = min(m, 2048)
        ref = reference(q[:n], cos[:n], sin[:n])
        score = lambda o: {"rel_max": float(np.abs(o[:n] - ref).max() / np.abs(ref).max())}
        row = {"tokens": m}

        def src_fn(q_, c_, s_):
            ec, es = dq.expanded_coefficients(c_, s_)  # the source's _expanded_coefficients
            return source(q_, ec, es, Hm, token_tile=64)

        ps_fn = lambda q_, c_, s_: dq.index_query_tail(q_, c_, s_, Hm, prefill=True)
        try:
            src = np.asarray(src_fn(jq, jc, js))
            row["source"] = {**score(src), **run(src_fn, (jq, jc, js), f"{m}_source")}
        except Exception as e:  # report, do not hide
            src = None
            row["source"] = {"error": repr(e)[:3000]}
        psj = np.asarray(ps_fn(jq, jc, js))
        row["ported_s_jax"] = {**score(psj), **run(ps_fn, (jq, jc, js), f"{m}_ported_s_jax")}
        pst = np.load(os.path.join(a.npz_dir, f"q_torch_{m}.npy"))
        tr = torch_rows[m]
        for name in ("ported_s", "target"):
            row[name] = {"us": tr[f"{name}_us"], "runs": tr[f"{name}_runs"],
                         **({"xprof": tr[f"{name}_xprof"]} if f"{name}_xprof" in tr else {})}
        row["ported_s"].update(score(pst))
        if src is not None:
            row["ported_s_jax_vs_source_exact"] = bool(np.array_equal(psj.view(np.int32), src.view(np.int32)))
            row["ported_s_vs_source_exact"] = bool(np.array_equal(pst.view(np.int32), src.view(np.int32)))
        print("Q " + json.dumps(_summary(row)), flush=True)
        report["rows"].append(row)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)


def _summary(row):
    out = {"tokens": row["tokens"]}
    for name in ("source", "ported_s_jax", "ported_s", "target"):
        r = row.get(name, {})
        if "error" in r:
            out[name] = {"error": r["error"][:300]}
            continue
        x = r.get("xprof", {})
        out[name] = {
            "wall_us": r.get("us"),
            "module_us": round(sum(u for _, u in x.get("modules", [])), 1) if "modules" in x else x.get("error"),
            "top_ops": [(o[:50], round(u, 1)) for o, u in x.get("ops", [])[:3]],
            "rel_max": r.get("rel_max"),
        }
    for k in ("ported_s_jax_vs_source_exact", "ported_s_vs_source_exact"):
        out[k] = row.get(k)
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
    p.add_argument("--out", default="/tmp/prologue3/q_report.json")
    a = p.parse_args()
    os.makedirs(a.npz_dir, exist_ok=True)
    stage_torch(a) if a.stage == "torch" else stage_jax(a)


if __name__ == "__main__":
    main()
