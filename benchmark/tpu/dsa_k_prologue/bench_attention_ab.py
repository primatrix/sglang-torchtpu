"""DSA sparse attention only (S5-S6): Ported-S torch op vs Target torch op.

Both sides get the same picked rows (seeded, 2048 per query, -1 padded): Ported-S
as flat cache slots (``topk_is_physical``), Target as the sequence-local rows its
indexer emits. Each timed call is the side's whole attention step as the backend
runs it: write this step's KV into the FP8 cache, then attend over the picked rows.

* ported_s: ``sglang_tpu_pallas::dsa_fp8_attention`` (split FP8, per-token f32
  scale, fp8-qk-pv), page 128.
* target: ``sglang_tpu_pallas::sparse_mla_attention_fused`` (per-query kernel, FP8
  per-tensor k_scale=1, the torch-side ``_quantize_kv_to_fp8`` in the graph). The
  PR 2198 baseline does not set SGLANG_TPU_ENABLE_QBLOCK_SPARSE_MLA_PREFILL.

Shapes: PR sgl-project/sglang-torchtpu#2198 baseline, per attention-DP rank (H=64).
``--stage prep`` (JAX on TPU) writes both prefilled caches to ``--npz-dir``;
``--stage torch`` times both ops (device module time, xprof) and scores each output
against f64 attention on the unquantized K/V over the same rows.
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

CASES = ("pr2198_extend_c1", "pr2198_extend_c2", "pr2198_verify_b4", "pr2198_decode_b4")
FP = att.FP


def stage_prep(a):
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.fp8_kv_cache import SplitFP8KVCache
    from sglang_tpu.attention.pallas.source_dsa.layers.attention.dsa_sparse_backend_helpers import (
        _scatter_fused_kv_paged,
    )

    for name in a.cases:
        c = att.case(name)
        P, n = c["P"], c["P"] * FP
        kv, pe = jnp.asarray(c["prefix_kv"]), jnp.asarray(c["prefix_pe"])
        # Ported-S: every slot holds a per-token-scaled prefix row, as the source writer stores it.
        empty = SplitFP8KVCache(jnp.zeros((2 * P, 64, 4, 128), jnp.uint8),
                                jnp.zeros((2 * P, 16, 4, 128), jnp.uint8))
        split = jax.jit(lambda cc, kv_, pe_: _scatter_fused_kv_paged(
            cc, kv_, pe_, jnp.array([n], jnp.int32), jnp.arange(2 * P, dtype=jnp.int32),
            jnp.array([0, n], jnp.int32), jnp.array([0, n], jnp.int32),
            kv_lora_rank=att.LAT, fp8_dtype=jnp.float8_e4m3fn))(empty, kv, pe)
        # Target: the same rows through its own writer (per-tensor k_scale=1).
        fill, _ = att.target_fns(c, qblock=False)
        tn, tr = jax.jit(fill, static_argnums=(4,))(
            jnp.zeros((P, FP, 4, 128), jnp.uint8), jnp.zeros((P, FP // 4, 4, 128), jnp.uint8),
            kv, pe, n, jnp.arange(P, dtype=jnp.int32))
        np.savez(os.path.join(a.npz_dir, f"att_{name}.npz"),
                 lat=np.asarray(split.latent).reshape(P, FP, 4, 128),
                 rope=np.asarray(split.rope).reshape(P, FP // 4, 4, 128),
                 tnope=np.asarray(tn), trope=np.asarray(tr))
        print(f"PREP {name} pages={P}", flush=True)
    return 0


def stage_torch(a):
    os.environ["JAX_PLATFORMS"] = "cpu"  # TorchTPU owns the device
    import torch
    import torch_tpu  # noqa: F401

    from bench_torch_ab import _profile
    from sglang_tpu.attention_pallas_mla_sparse import _build_pallas_sparse_mla_kernel_op
    from sglang_tpu.attention_pallas_rpa_base import _quantize_kv_to_fp8
    from sglang_tpu.dsa_fp8_attention_op import build_dsa_fp8_attention_op

    # Registered outside torch.compile: Dynamo cannot trace op registration.
    pop, top = build_dsa_fp8_attention_op(), _build_pallas_sparse_mla_kernel_op()
    sync = torch.accelerator.synchronize
    dev = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to("tpu")
    bf = lambda x: torch.from_numpy(np.ascontiguousarray(x).view(np.int16)).view(
        torch.bfloat16).to("tpu")
    comp = lambda f: torch.compile(f, backend="tpu", fullgraph=True, dynamic=False)
    f8 = torch.float8_e4m3fn

    for name in a.cases:
        c = att.case(name)
        ref = np.load(os.path.join(a.npz_dir, f"att_{name}.npz"))
        S, prefill = c["S"], c["mode"] == "prefill"
        sm = float(c["sm_scale"])
        rows_bf = att.bf16_rows(c)
        ql, qr, kvc, kpe = (bf(c[k]) for k in ("q_latent", "q_rope", "kv_c", "k_pe"))
        kv, pi, cuq = dev(c["kv"]), dev(c["pi"]), dev(c["cu_q"])
        slots = dev(to_physical(c["topk"], c["pi"], c["ppr"], c["cu_q"]))
        logical = dev(c["topk"])
        cukv = dev((np.arange(S + 1) * c["ppr"] * FP).astype(np.int32))
        dist = dev(np.array([S] * 3 if c["mode"] == "decode" else [0, 0, S], np.int32))

        def ported(lat, rope, ql_, qr_, kvc_, kpe_):
            return pop(lat, rope, ql_, qr_, kvc_, kpe_, slots, kv, pi, cuq, sm, True, prefill,
                       True)

        def target(nope, rope, ql_, qr_, kvc_, kpe_):
            return top(ql_, qr_, _quantize_kv_to_fp8(kvc_, 1.0, f8),
                       _quantize_kv_to_fp8(kpe_, 1.0, f8), nope, rope, kv, pi, cuq, cukv, dist,
                       sm, 1.0, 1.0, logical)

        row = {"case": name, "T": c["T"], "mode": c["mode"]}
        for side, fn, caches in (("ported_s", ported, ("lat", "rope")),
                                 ("target", target, ("tnope", "trope"))):
            try:
                cf = comp(fn)
                fresh = lambda: tuple(dev(ref[k]) for k in caches)
                out = cf(*fresh(), ql, qr, kvc, kpe)
                out = out.cpu().view(torch.int16).numpy().view(np.uint16)
                r ={"accuracy_vs_bf16": att.error_vs_bf16(c, _as_f64(out), rows_bf)}
                bargs = (*fresh(), ql, qr, kvc, kpe)
                r["us"], _ = att._median_time(lambda: cf(*bargs), sync, a.iters, a.repeats)
                if a.xprof_dir:
                    prof = _profile(cf, (*fresh(), ql, qr, kvc, kpe), a.iters,
                                    f"{a.xprof_dir}/att_{side}_{name}")
                    r["module_us"] = round(sum(u for _, u in prof.get("modules", [])), 1)
                    r["top_ops"] = [(n[:60], round(u, 1)) for n, u in prof.get("ops", [])[:8]]
            except Exception as e:  # report, do not hide; the other side still runs
                r = {"error": repr(e)[:2000]}
            row[side] = r
        print("ATT " + json.dumps(row), flush=True)
    return 0


def _as_f64(u16):
    """bf16 bit patterns (uint16) -> float64."""
    return (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("prep", "torch"), required=True)
    p.add_argument("--cases", nargs="+", default=list(CASES))
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--npz-dir", required=True)
    p.add_argument("--xprof-dir", default="")
    a = p.parse_args()
    os.makedirs(a.npz_dir, exist_ok=True)
    sys.exit(stage_prep(a) if a.stage == "prep" else stage_torch(a))


if __name__ == "__main__":
    main()
