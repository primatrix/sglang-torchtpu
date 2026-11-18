"""Test only: #2246 baseline vs #2246 + all optimizations, through the torch op.

Two source trees, each run in its own processes with its tree first on
PYTHONPATH (``--tree base|full``; the pod checks them out side by side):

* ``base``: #2246 head (ac230df), SparseCore cache with 128 B RoPE rows;
* ``full``: + packed 64 B RoPE cache, fixed A gather (phase stride 5, compact
  RoPE, 32-row DMA, 32 atoms), 512-wide kernel output and split q.

``--stage prep`` (JAX on TPU) fills the tree's own SparseCore cache layout with
the same FP8 rows; ``--stage time`` runs
``sglang_tpu_pallas::sparse_mla_attention_fused_phys`` under torch.compile on it
(step = torch-side FP8 quantize + KV write + attention), saves the outputs and,
on layer 0, profiles 10 calls and checks the body (base keeps the 640->512 output
slice and the q concat/pad; full has neither). ``--stage sum`` compares outputs
bit for bit and prints the medians. Prints ``FAB`` lines.
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
from check_physical_topk import to_physical  # noqa: E402

FP = att.FP = 64
CASE = "pr2198_extend_c1"
SAMPLES = ("synthetic", "raw-0096-layer0", "raw-0097-layer10", "raw-0099-layer40")


def _src(a):
    import sglang_tpu

    return os.path.dirname(os.path.dirname(os.path.abspath(sglang_tpu.__file__)))


def stage_prep(a):
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    from sglang_tpu.attention.pallas.mla_sparse import kv_cache_utils as kcu

    c = att.case(CASE)
    P, n = c["P"], c["P"] * FP
    lay = kcu.KVCacheLayout.SPARSECORE
    ns = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.NOPE, lay, P, FP, 512)
    rs = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.ROPE, lay, P, FP, 64)
    f8 = jnp.float8_e4m3fn
    fmax = float(jnp.finfo(f8).max)
    q8 = lambda x: jnp.clip(x.astype(jnp.float32), -fmax, fmax).astype(f8)
    sn, sr = jax.jit(lambda no, ro, kv_, pe_: kcu.update_sparse_mla_kv_cache_jax(
        no, ro, q8(kv_), q8(pe_), jnp.array([n], jnp.int32), jnp.arange(P, dtype=jnp.int32),
        jnp.array([0, n], jnp.int32), nope_spec=ns, rope_spec=rs))(
        jnp.zeros(ns.shape, ns.jax_dtype), jnp.zeros(rs.shape, rs.jax_dtype),
        jnp.asarray(c["prefix_kv"]), jnp.asarray(c["prefix_pe"]))
    np.savez(os.path.join(a.npz_dir, f"cache_{a.tree}.npz"), nope=np.asarray(sn),
             rope=np.asarray(sr))
    print("FAB " + json.dumps({"stage": "prep", "tree": a.tree, "src": _src(a),
                               "nope": list(sn.shape), "rope": list(sr.shape)}), flush=True)
    return 0


def stage_time(a):
    os.environ["JAX_PLATFORMS"] = "cpu"  # TorchTPU owns the device
    import torch
    import torch._dynamo
    import torch_tpu  # noqa: F401

    from sglang_tpu.attention_pallas_mla_sparse import _build_pallas_sparse_mla_kernel_op
    from sglang_tpu.attention_pallas_rpa_base import _quantize_kv_to_fp8

    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 512
    op = _build_pallas_sparse_mla_kernel_op(topk_is_physical=True)
    sync = torch.accelerator.synchronize
    dev = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to("tpu")
    bf = lambda x: torch.from_numpy(np.ascontiguousarray(x).view(np.int16)).view(
        torch.bfloat16).to("tpu")
    f8 = torch.float8_e4m3fn

    c = att.case(CASE)
    ref = np.load(os.path.join(a.npz_dir, f"cache_{a.tree}.npz"))
    S, sm = c["S"], float(c["sm_scale"])
    ql, qr, kvc, kpe = (bf(c[k]) for k in ("q_latent", "q_rope", "kv_c", "k_pe"))
    kv, pi, cuq = dev(c["kv"]), dev(c["pi"]), dev(c["cu_q"])
    cukv = dev((np.arange(S + 1) * c["ppr"] * FP).astype(np.int32))
    dist = dev(np.array([0, 0, S], np.int32))
    paths = {os.path.basename(f)[:-4]: f
             for f in glob.glob(f"{a.raw_dir}/**/raw-*.npy", recursive=True)}
    for name in SAMPLES:
        r = {"stage": "time", "tree": a.tree, "pass": a.pass_id, "sample": name}
        if name == SAMPLES[0]:
            r["src"] = _src(a)
        try:
            lg = (np.asarray(c["topk"]) if name == "synthetic"
                  else np.load(paths[name])).astype(np.int32)
            topk = dev(to_physical(lg, c["pi"], c["ppr"], c["cu_q"], page=FP))

            def fn(nope, rope, ql_, qr_, kvc_, kpe_, _topk=topk):
                return op(ql_, qr_, _quantize_kv_to_fp8(kvc_, 1.0, f8),
                          _quantize_kv_to_fp8(kpe_, 1.0, f8), nope, rope, kv, pi, cuq,
                          cukv, dist, sm, 1.0, 1.0, _topk)

            torch._dynamo.reset()
            cf = torch.compile(fn, backend="tpu", fullgraph=True, dynamic=False)
            nope, rope = dev(ref["nope"]), dev(ref["rope"])
            out = cf(nope, rope, ql, qr, kvc, kpe)
            sync()
            o = out.cpu().view(torch.int16).numpy().view(np.uint16)
            r["shape"] = list(o.shape)
            np.save(os.path.join(a.npz_dir, f"out_{a.tree}_{name}_p{a.pass_id}.npy"), o)
            bargs = (nope, rope, ql, qr, kvc, kpe)
            med, runs = att._median_time(lambda: cf(*bargs), sync, a.iters, a.repeats)
            r.update(us=round(med, 1), us_range=[round(min(runs), 1), round(max(runs), 1)])
            if a.pass_id == 0 and name == "raw-0096-layer0":
                from bench_torch_ab import _profile

                s = _profile(cf, bargs, 10, os.path.join(a.npz_dir, f"xprof_{a.tree}"))
                if "error" in s:
                    r["prof_error"] = s["error"][:400]
                else:
                    names = [n for n, _ in s["ops"]]
                    has_slice = any(n.startswith("slice.") or
                                    ("bf16[2048,64,512]" in n and " slice(" in n)
                                    for n in names)
                    has_qcat = any("fusion.2.q_hbm_ref" in n for n in names)
                    expect = (True, True) if a.tree == "base" else (False, False)
                    r.update(out_slice=has_slice, q_concat=has_qcat,
                             body_ok=(has_slice, has_qcat) == expect,
                             module_us=round(sum(u for _, u in s.get("modules", [])), 1))
                    for n, u in s["ops"][:12]:
                        print("FABOP " + json.dumps({"tree": a.tree, "us": round(u, 1),
                                                     "op": n[:140]}), flush=True)
        except Exception as e:  # report, do not hide
            import traceback

            traceback.print_exc()
            r["fail"] = repr(e)[:600]
        print("FAB " + json.dumps(r), flush=True)
        with open(os.path.join(a.npz_dir, f"time_{a.tree}_p{a.pass_id}.jsonl"), "a") as f:
            f.write(json.dumps(r) + "\n")
    return 0


def stage_sum(a):
    for name in SAMPLES:
        s = {"stage": "sum", "sample": name, "us": {}, "passes": {}}
        for t in ("base", "full"):
            us = []
            for f in sorted(glob.glob(os.path.join(a.npz_dir, f"time_{t}_p*.jsonl"))):
                for line in open(f):
                    r = json.loads(line)
                    if r.get("sample") == name and "us" in r:
                        us.append(r["us"])
            if us:
                s["passes"][t] = us
                s["us"][t] = round(sum(us) / len(us), 1)
        x = os.path.join(a.npz_dir, f"out_base_{name}_p0.npy")
        y = os.path.join(a.npz_dir, f"out_full_{name}_p0.npy")
        if os.path.exists(x) and os.path.exists(y):
            bx, by = np.load(x), np.load(y)
            s["exact"] = bool(bx.shape == by.shape and np.array_equal(bx, by))
            if not s["exact"] and bx.shape == by.shape:
                fx = (bx.astype(np.uint32) << 16).view(np.float32)
                fy = (by.astype(np.uint32) << 16).view(np.float32)
                s["max_abs"] = float(np.nanmax(np.abs(fx - fy)))
        if len(s["us"]) == 2:
            s["saved_us"] = round(s["us"]["base"] - s["us"]["full"], 1)
            s["saved_frac"] = round(s["saved_us"] / s["us"]["base"], 4)
        print("FAB_SUM " + json.dumps(s), flush=True)
    print("FAB_DONE", flush=True)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("prep", "time", "sum"), required=True)
    p.add_argument("--tree", choices=("base", "full"))
    p.add_argument("--pass-id", type=int, default=0)
    p.add_argument("--npz-dir", required=True)
    p.add_argument("--raw-dir", default="/old/fj8")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--repeats", type=int, default=10)
    a = p.parse_args()
    os.makedirs(a.npz_dir, exist_ok=True)
    sys.exit({"prep": stage_prep, "time": stage_time, "sum": stage_sum}[a.stage](a))


if __name__ == "__main__":
    main()
