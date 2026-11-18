"""Physical top-k between the migrated indexer (S1-S4) and attention (S5-S6) torch ops.

Source deployment: the indexer emits flat cache slots (``output_physical_slots``)
and the attention only validates them (``topk_indices_are_physical``), so the
per-layer logical-to-physical gather (25.7 ms at extend 2048) is not paid.

1. ``dsa_bf16_indexer`` with ``physical=True`` must return, row for row, the
   page-table mapping of its ``physical=False`` output (slot = page * 128 + offset,
   -1 kept), and store the same cache bytes.
2. ``dsa_fp8_attention`` with ``topk_is_physical=True`` on those mapped slots must
   return the logical path's output byte for byte (the Ported-S JAX npz from
   ``bench_fp8_attention_ab.py --npz-dir``), and is timed against it.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_fp8_attention_ab as att  # noqa: E402
import bench_indexer_chain as bc  # noqa: E402


def to_physical(logical, pi, ppr, cu_q, page=128):
    """Sequence-local positions -> flat framework slots; -1 stays -1."""
    seq = np.searchsorted(cu_q[1:], np.arange(logical.shape[0]), side="right")
    lg = np.maximum(logical, 0)
    pages = pi[seq[:, None] * ppr + lg // page]
    return np.where(logical >= 0, pages * page + lg % page, -1).astype(np.int32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz-dir", required=True)
    p.add_argument("--xprof-dir", default="")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()

    os.environ["JAX_PLATFORMS"] = "cpu"  # TorchTPU owns the device
    import torch
    import torch_tpu  # noqa: F401

    from bench_torch_ab import _profile
    from sglang_tpu.dsa_bf16_indexer_op import build_dsa_bf16_indexer_op
    from sglang_tpu.dsa_fp8_attention_op import build_dsa_fp8_attention_op

    sync = torch.accelerator.synchronize
    comp = lambda f: torch.compile(f, backend="tpu", fullgraph=True, dynamic=False)
    t = lambda x: bc._to_torch(torch, x)
    ok = True

    iop = build_dsa_bf16_indexer_op()
    H = torch.from_numpy(bc.hadamard()).to("tpu")
    for name in ("extend_2048", "extend_unaligned", "decode_b64"):
        c = bc.case(name)
        ins = tuple(t(c[k]) for k in ("hidden", "wk", "ws", "nw", "nb", "query", "cos", "sin",
                                      "w", "kv", "pi", "cu_q", "dist")) + (H,)
        res = {}
        for phys in (False, True):
            cache = t(c["cache"])
            fn = comp(lambda cc, *x, _p=phys: iop(cc, *x, bc.K, c["prefill"], c["decode"], _p))
            rows = fn(cache, *ins).cpu().numpy()
            res[phys] = (rows, cache.cpu().view(torch.int16).numpy())
        mapped = to_physical(res[False][0], c["pi"], c["ppr"], c["cu_q"])
        row = {
            "case": name,
            "rows_equal_mapped": bool(np.array_equal(res[True][0], mapped)),
            "sets_equal_mapped": bool(np.array_equal(np.sort(res[True][0], 1), np.sort(mapped, 1))),
            "cache_equal": bool(np.array_equal(res[True][1], res[False][1])),
            "valid_per_row_min": int((mapped >= 0).sum(1).min()),
        }
        ok &= row["sets_equal_mapped"] and row["cache_equal"]
        print("PHYSIDX " + json.dumps(row), flush=True)

    aop = build_dsa_fp8_attention_op()
    bf = lambda x: torch.from_numpy(np.ascontiguousarray(x).view(np.int16)).view(torch.bfloat16).to("tpu")
    i32 = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to("tpu")
    u8 = i32
    for name in ("extend_2048", "extend_unaligned", "decode_b64"):
        c = att.case(name)
        ref = np.load(os.path.join(a.npz_dir, f"fp8att_{name}.npz"))
        P = c["P"]
        lat0 = ref["lat0"].reshape(P, att.FP, 4, 128)
        rope0 = ref["rope0"].reshape(P, att.FP // 4, 4, 128)
        prefill = c["mode"] == "prefill"
        phys = to_physical(c["topk"], c["pi"], c["ppr"], c["cu_q"])
        row = {"case": name}
        for tag, topk, is_phys in (("logical", c["topk"], False), ("physical", phys, True)):
            static = (float(c["sm_scale"]), True, prefill, is_phys)
            ins = (bf(c["q_latent"]), bf(c["q_rope"]), bf(c["kv_c"]), bf(c["k_pe"]), i32(topk),
                   i32(c["kv"]), i32(c["pi"]), i32(c["cu_q"]))
            fn = comp(lambda lat, rope, *x, _s=static: aop(lat, rope, *x, *_s))
            lat, rope = u8(lat0), u8(rope0)
            out = fn(lat, rope, *ins).cpu().view(torch.int16).numpy().view(np.uint16)
            sync()
            r = {"out_exact_vs_jax": bool(np.array_equal(out, ref["out"])),
                 "cache_exact_vs_jax": bool(
                     np.array_equal(lat.cpu().numpy().reshape(-1), ref["latent"].reshape(-1))
                     and np.array_equal(rope.cpu().numpy().reshape(-1), ref["rope"].reshape(-1)))}
            bl, br = u8(lat0), u8(rope0)
            r["us"], _ = att._median_time(lambda: fn(bl, br, *ins), sync, a.iters, a.repeats)
            if a.xprof_dir:
                prof = _profile(fn, (u8(lat0), u8(rope0), *ins), a.iters,
                                f"{a.xprof_dir}/phys_{tag}_{name}")
                r["module_us"] = round(sum(u for _, u in prof.get("modules", [])), 1)
                r["top_ops"] = [(n[:60], round(u, 1)) for n, u in prof.get("ops", [])[:4]]
            ok &= r["out_exact_vs_jax"] and r["cache_exact_vs_jax"]
            row[tag] = r
        print("PHYSATT " + json.dumps(row), flush=True)
    print("PHYS_ALL_OK" if ok else "PHYS_FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
