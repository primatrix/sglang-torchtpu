"""Test only: fixed A / A + narrow / A + narrow + split q through the torch op.

The jax.jit A/B (bench_a_narrow_ab.py, exp-woy7i7oix8) saved ~0.18-0.19 ms per
step. narrow_output alone showed no gain through the torch op before
(exp-10e3m2l6i4), so this times the production path:
``sglang_tpu_pallas::sparse_mla_attention_fused_phys`` under torch.compile on the
SparseCore-layout cache, step = torch-side FP8 quantize + KV write + attention.

``DSA_TORCH_AB`` (read at trace time in mla_sparse_fused.py) picks the variant;
the op is registered once per process under one name, so each variant runs in
its own process (``--variant``). ``run_all`` alternates A / A_narrow / A_splitq
twice, each a fresh process on the same prefilled cache (bench_target_baseline
``--stage prep``). Outputs are saved and compared to A bit for bit. Prints
``TAB`` lines.
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_fp8_attention_ab as att  # noqa: E402
from check_physical_topk import to_physical  # noqa: E402

FP = att.FP = 64
CASE = "pr2198_extend_c1"
SAMPLES = ("synthetic", "raw-0096-layer0", "raw-0097-layer10", "raw-0099-layer40")
VARIANTS = ("A", "A_narrow", "A_splitq")


def run_variant(a):
    os.environ["JAX_PLATFORMS"] = "cpu"  # TorchTPU owns the device
    os.environ["DSA_TORCH_AB"] = a.variant
    import torch
    import torch_tpu  # noqa: F401
    import torch._dynamo

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
    ref = np.load(os.path.join(a.npz_dir, f"base_{CASE}.npz"))
    S, sm = c["S"], float(c["sm_scale"])
    ql, qr, kvc, kpe = (bf(c[k]) for k in ("q_latent", "q_rope", "kv_c", "k_pe"))
    kv, pi, cuq = dev(c["kv"]), dev(c["pi"]), dev(c["cu_q"])
    cukv = dev((np.arange(S + 1) * c["ppr"] * FP).astype(np.int32))
    dist = dev(np.array([0, 0, S], np.int32))
    paths = {os.path.basename(f)[:-4]: f
             for f in glob.glob(f"{a.raw_dir}/**/raw-*.npy", recursive=True)}
    for name in SAMPLES:
        r = {"variant": a.variant, "pass": a.pass_id, "sample": name}
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
            nope, rope = dev(ref["sc_nope"]), dev(ref["sc_rope"])
            out = cf(nope, rope, ql, qr, kvc, kpe)
            sync()
            o = out.cpu().view(torch.int16).numpy().view(np.uint16)
            r["shape"] = list(o.shape)
            np.save(os.path.join(a.npz_dir, f"out_{a.variant}_{name}_p{a.pass_id}.npy"), o)
            bargs = (nope, rope, ql, qr, kvc, kpe)
            med, runs = att._median_time(lambda: cf(*bargs), sync, a.iters, a.repeats)
            r.update(us=round(med, 1), us_range=[round(min(runs), 1), round(max(runs), 1)])
            if a.profile and name == "raw-0096-layer0":
                # xprof of 10 warm calls: per-op self time and per-module time,
                # so ops the torch boundary adds around the custom call show up.
                from bench_torch_ab import _profile

                pd = os.path.join(a.npz_dir, f"xprof_{a.variant}")
                s = _profile(cf, bargs, 10, pd)
                if "error" in s:
                    print("TPROF " + json.dumps({"variant": a.variant, "error": s["error"][:500]}),
                          flush=True)
                else:
                    print("TPROF " + json.dumps({
                        "variant": a.variant, "device_us": round(s["device_us"], 1),
                        "modules": [(n[:60], round(u, 1)) for n, u in s["modules"][:8]]}),
                        flush=True)
                    # One op per line so no line is cut by the log filter.
                    for n, u in s["ops"][:25]:
                        print("TPROFOP " + json.dumps({"v": a.variant, "us": round(u, 1),
                                                       "op": n[:150]}), flush=True)
        except Exception as e:  # report, do not hide
            import traceback

            traceback.print_exc()
            r["fail"] = repr(e)[:600]
        print("TAB " + json.dumps(r), flush=True)
    return 0


def run_all(a):
    order = [VARIANTS, VARIANTS[::-1]]
    rows = []
    for p, seq in enumerate(order):
        for v in seq:
            cmd = [sys.executable, "-u", __file__, "--variant", v, "--pass-id", str(p),
                   "--npz-dir", a.npz_dir, "--raw-dir", a.raw_dir,
                   "--iters", str(a.iters), "--repeats", str(a.repeats)]
            if a.profile and p == 0:
                cmd.append("--profile")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            for line in proc.stdout.splitlines():
                if line.startswith(("TPROF", "TPROFOP")):
                    print(line, flush=True)
                if line.startswith("TAB "):
                    print(line, flush=True)
                    rows.append(json.loads(line[4:]))
            if proc.returncode:
                print("TAB " + json.dumps({"variant": v, "pass": p, "rc": proc.returncode,
                                           "stderr_tail": proc.stderr[-1500:]}), flush=True)
    # Summary per sample: median of the two passes, exactness vs A.
    for name in SAMPLES:
        s = {"sample": name, "us": {}, "exact_vs_A": {}}
        for v in VARIANTS:
            us = [r["us"] for r in rows if r.get("sample") == name and r.get("variant") == v
                  and "us" in r]
            if us:
                s["us"][v] = round(sum(us) / len(us), 1)
                s.setdefault("passes", {})[v] = us
        base = os.path.join(a.npz_dir, f"out_A_{name}_p0.npy")
        for v in VARIANTS[1:]:
            f = os.path.join(a.npz_dir, f"out_{v}_{name}_p0.npy")
            if os.path.exists(base) and os.path.exists(f):
                x, y = np.load(base), np.load(f)
                if x.shape != y.shape and x.shape[-1] > y.shape[-1]:
                    x = x[..., :y.shape[-1]]
                s["exact_vs_A"][v] = bool(x.shape == y.shape and np.array_equal(x, y))
        if "A" in s["us"]:
            s["saved_us"] = {v: round(s["us"]["A"] - s["us"][v], 1)
                             for v in VARIANTS[1:] if v in s["us"]}
        print("TAB_SUM " + json.dumps(s), flush=True)
    print("TAB_DONE", flush=True)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=VARIANTS)
    p.add_argument("--pass-id", type=int, default=0)
    p.add_argument("--npz-dir", required=True)
    p.add_argument("--raw-dir", default="/old/fj8")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--profile", action="store_true")  # xprof on layer 0, pass 0
    a = p.parse_args()
    os.makedirs(a.npz_dir, exist_ok=True)
    sys.exit(run_variant(a) if a.variant else run_all(a))


if __name__ == "__main__":
    main()
