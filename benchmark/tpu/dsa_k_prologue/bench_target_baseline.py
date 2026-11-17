"""DSA sparse attention (S5-S6) baseline: Target torch op and its variants, PR 2198 shapes.

Every variant runs in the same process on the same inputs (page 64, seeded Q / KV /
top-k, 2048 picked rows per query, -1 padded), so each one is compared with the
unmodified Target measured in the same run:

* ``target``: ``sglang_tpu_pallas::sparse_mla_attention_fused`` as the PR 2198
  baseline serves it (per-query kernel, sequence-local rows, FP8 per-tensor
  k_scale=1, the torch-side ``_quantize_kv_to_fp8`` in the graph).
* ``target_phys``: the same op with ``topk_is_physical=True``: flat cache slots in,
  no page-table lookup (``gather_page_ids``).

Each timed call writes this step's KV into the cache, then attends. Reported per
variant: median wall time over ``--repeats`` runs and its range, device module time
(xprof), top ops, and the error vs f64 attention on the unquantized K/V over the same
rows. ``--stage prep`` (JAX on TPU) fills the cache; ``--stage torch`` times.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_fp8_attention_ab as att  # noqa: E402
from check_physical_topk import to_physical  # noqa: E402

FP = att.FP = 64  # the PR 2198 baseline page size
CASES = ("pr2198_extend_c1", "pr2198_extend_c2", "pr2198_verify_b4", "pr2198_decode_b4")


def stage_prep(a):
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    for name in a.cases:
        c = att.case(name)
        P, n = c["P"], c["P"] * FP
        fill, _ = att.target_fns(c, qblock=False)
        tn, tr = jax.jit(fill, static_argnums=(4,))(
            jnp.zeros((P, FP, 4, 128), jnp.uint8), jnp.zeros((P, FP // 4, 4, 128), jnp.uint8),
            jnp.asarray(c["prefix_kv"]), jnp.asarray(c["prefix_pe"]), n,
            jnp.arange(P, dtype=jnp.int32))
        np.savez(os.path.join(a.npz_dir, f"base_{name}.npz"), nope=np.asarray(tn),
                 rope=np.asarray(tr))
        print(f"PREP {name} pages={P}", flush=True)
    return 0


def stage_torch(a):
    os.environ["JAX_PLATFORMS"] = "cpu"  # TorchTPU owns the device
    import torch
    import torch_tpu  # noqa: F401

    from bench_torch_ab import _profile
    from sglang_tpu.attention_pallas_mla_sparse import _build_pallas_sparse_mla_kernel_op
    from sglang_tpu.attention_pallas_rpa_base import _quantize_kv_to_fp8

    # Registered outside torch.compile: Dynamo cannot trace op registration.
    ops = {"target": _build_pallas_sparse_mla_kernel_op(),
           "target_phys": _build_pallas_sparse_mla_kernel_op(topk_is_physical=True)}
    sync = torch.accelerator.synchronize
    dev = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to("tpu")
    bf = lambda x: torch.from_numpy(np.ascontiguousarray(x).view(np.int16)).view(
        torch.bfloat16).to("tpu")
    comp = lambda f: torch.compile(f, backend="tpu", fullgraph=True, dynamic=False)
    f8 = torch.float8_e4m3fn

    for name in a.cases:
        c = att.case(name)
        ref = np.load(os.path.join(a.npz_dir, f"base_{name}.npz"))
        S = c["S"]
        sm = float(c["sm_scale"])
        rows_bf = att.bf16_rows(c)
        ql, qr, kvc, kpe = (bf(c[k]) for k in ("q_latent", "q_rope", "kv_c", "k_pe"))
        kv, pi, cuq = dev(c["kv"]), dev(c["pi"]), dev(c["cu_q"])
        rows = {"target": dev(c["topk"]),
                "target_phys": dev(to_physical(c["topk"], c["pi"], c["ppr"], c["cu_q"], page=FP))}
        cukv = dev((np.arange(S + 1) * c["ppr"] * FP).astype(np.int32))
        dist = dev(np.array([S] * 3 if c["mode"] == "decode" else [0, 0, S], np.int32))
        row = {"case": name, "T": c["T"], "mode": c["mode"]}
        outs = {}
        for side, op in ops.items():
            topk = rows[side]

            def fn(nope, rope, ql_, qr_, kvc_, kpe_, _op=op, _topk=topk):
                return _op(ql_, qr_, _quantize_kv_to_fp8(kvc_, 1.0, f8),
                           _quantize_kv_to_fp8(kpe_, 1.0, f8), nope, rope, kv, pi, cuq, cukv,
                           dist, sm, 1.0, 1.0, _topk)

            try:
                cf = comp(fn)
                fresh = lambda: (dev(ref["nope"]), dev(ref["rope"]))
                out = cf(*fresh(), ql, qr, kvc, kpe).cpu().view(torch.int16).numpy()
                outs[side] = out
                r = {"accuracy_vs_bf16": att.error_vs_bf16(c, _as_f64(out.view(np.uint16)),
                                                           rows_bf)}
                bargs = (*fresh(), ql, qr, kvc, kpe)
                med, runs = att._median_time(lambda: cf(*bargs), sync, a.iters, a.repeats)
                r["us"] = round(med, 1)
                r["us_range"] = [round(min(runs), 1), round(max(runs), 1)]
                if a.xprof_dir:
                    prof = _profile(cf, (*fresh(), ql, qr, kvc, kpe), a.iters,
                                    f"{a.xprof_dir}/base_{side}_{name}")
                    r["module_us"] = round(sum(u for _, u in prof.get("modules", [])), 1)
                    r["top_ops"] = [(n[:60], round(u, 1)) for n, u in prof.get("ops", [])[:8]]
            except Exception as e:  # report, do not hide; the other variants still run
                r = {"error": repr(e)[:2000]}
            row[side] = r
        if "target" in outs and "target_phys" in outs:
            # Same rows, same cache: the variant must not change the output.
            row["target_phys_out_exact"] = bool(np.array_equal(outs["target"], outs["target_phys"]))
        print("BASE " + json.dumps(row), flush=True)
    print("BASE_DONE", flush=True)
    return 0


def _as_f64(u16):
    """bf16 bit patterns (uint16) -> float64."""
    return (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("prep", "torch"), required=True)
    p.add_argument("--cases", nargs="+", default=list(CASES))
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--npz-dir", required=True)
    p.add_argument("--xprof-dir", default="")
    a = p.parse_args()
    os.makedirs(a.npz_dir, exist_ok=True)
    sys.exit(stage_prep(a) if a.stage == "prep" else stage_torch(a))


if __name__ == "__main__":
    main()
