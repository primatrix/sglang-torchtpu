"""Why the SparseCore-layout Target is slow through the torch op (exp-42iw4cidov).

At extend 2048 the torch op on a uint32 SparseCore cache took ~23.2 ms, while the
sweep's direct JAX call of the same kernel settings took ~3.6 ms (exp-cdkg2zm04b).
This runs ONE variant per process (the profiler only captured the first variant of
a multi-variant process) on the baseline bench's prepared caches:

* ``jax_tc`` / ``jax_sc``: ``mla_sparse_fused.sparse_mla_attention_fused`` under
  ``jax.jit``, physical slots, caches donated, TensorCore (uint8) / SparseCore
  (uint32) layout.
* ``torch_tc`` / ``torch_sc``: the torch op (``_build_pallas_sparse_mla_kernel_op(
  topk_is_physical=True)``) under ``torch.compile``, same caches, as
  bench_target_baseline runs it.

Prints one ``DIAG`` line: wall time, device module time, the largest ops, the
compiled modules, and the output's checksum (all four must agree).
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

FP = att.FP = 64


def _summ(d, iters):
    from xplane_summary import summarize

    x = summarize(d, iters, top=15)
    return {"module_us": round(sum(u for _, u in x.get("modules", [])), 1),
            "modules": [(n[:50], round(u, 1)) for n, u in x.get("modules", [])[:6]],
            "top_ops": [(n[:70], round(u, 1)) for n, u in x.get("ops", [])[:15]],
            "summary_error": x.get("error", "")}


def run_jax(a, c, ref, layout):
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    from sglang_tpu.attention.pallas.mla_sparse_fused import sparse_mla_attention_fused

    f8 = jnp.float8_e4m3fn
    fmax = float(jnp.finfo(f8).max)
    q8 = lambda x: jnp.clip(x.astype(jnp.float32), -fmax, fmax).astype(f8)
    S = c["S"]
    keys = ("sc_nope", "sc_rope") if layout == "sc" else ("nope", "rope")
    phys = jnp.asarray(to_physical(c["topk"], c["pi"], c["ppr"], c["cu_q"], page=FP))
    kvl, pi, cuq = (jnp.asarray(c[k]) for k in ("kv", "pi", "cu_q"))
    dist = jnp.asarray(np.array([S] * 3 if c["mode"] == "decode" else [0, 0, S], np.int32))
    ins = tuple(jnp.asarray(c[k]) for k in ("q_latent", "q_rope", "kv_c", "k_pe"))
    sm = float(c["sm_scale"])

    def step(no, ro, ql, qr, kvc, kpe):
        return sparse_mla_attention_fused(ql, qr, q8(kvc), q8(kpe), no, ro, kvl, pi, cuq, dist,
                                          phys, sm_scale=sm, k_scale=1.0, topk_is_physical=True)

    jf = jax.jit(step, donate_argnums=(0, 1))
    st = {"c": (jnp.asarray(ref[keys[0]]), jnp.asarray(ref[keys[1]]))}

    def call():
        out, no, ro = jf(*st["c"], *ins)
        st["c"] = (no, ro)
        return out

    hlo_out = os.environ.get("HLO_OUT")
    if hlo_out:
        # Fallback to XLA_FLAGS dumps: the compiled (after optimizations) module text.
        try:
            text = jf.lower(*st["c"], *ins).compile().as_text()
            os.makedirs(hlo_out, exist_ok=True)
            with open(os.path.join(hlo_out, "jax_step.after_optimizations.txt"), "w") as f:
                f.write(text)
        except Exception as e:  # report, do not hide
            print(f"HLO_TEXT_ERROR {e!r}"[:500], flush=True)
    out = np.asarray(call())
    sync = lambda: jax.block_until_ready(st["c"][0])
    r = {"us": round(att._median_time(call, sync, a.iters, a.repeats)[0], 1)}
    d = f"{a.xprof_dir}/diag_{a.variant}_{a.case}"
    sync()
    jax.profiler.start_trace(d)
    for _ in range(a.iters):
        call()
    sync()
    jax.profiler.stop_trace()
    r.update(_summ(d, a.iters))
    return r, out.view(np.uint16)


def run_torch(a, c, ref, layout):
    os.environ["JAX_PLATFORMS"] = "cpu"
    import torch
    import torch_tpu  # noqa: F401

    from bench_torch_ab import _profile  # noqa: F401  (same capture as the baseline)
    from sglang_tpu.attention_pallas_mla_sparse import _build_pallas_sparse_mla_kernel_op
    from sglang_tpu.attention_pallas_rpa_base import _quantize_kv_to_fp8

    op = _build_pallas_sparse_mla_kernel_op(topk_is_physical=True)
    sync = torch.accelerator.synchronize
    dev = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to("tpu")
    bf = lambda x: torch.from_numpy(np.ascontiguousarray(x).view(np.int16)).view(
        torch.bfloat16).to("tpu")
    f8 = torch.float8_e4m3fn
    S = c["S"]
    keys = ("sc_nope", "sc_rope") if layout == "sc" else ("nope", "rope")
    ql, qr, kvc, kpe = (bf(c[k]) for k in ("q_latent", "q_rope", "kv_c", "k_pe"))
    kv, pi, cuq = dev(c["kv"]), dev(c["pi"]), dev(c["cu_q"])
    phys = dev(to_physical(c["topk"], c["pi"], c["ppr"], c["cu_q"], page=FP))
    cukv = dev((np.arange(S + 1) * c["ppr"] * FP).astype(np.int32))
    dist = dev(np.array([S] * 3 if c["mode"] == "decode" else [0, 0, S], np.int32))
    sm = float(c["sm_scale"])

    def fn(nope, rope, ql_, qr_, kvc_, kpe_):
        return op(ql_, qr_, _quantize_kv_to_fp8(kvc_, 1.0, f8), _quantize_kv_to_fp8(kpe_, 1.0, f8),
                  nope, rope, kv, pi, cuq, cukv, dist, sm, 1.0, 1.0, phys)

    cf = torch.compile(fn, backend="tpu", fullgraph=True, dynamic=False)
    nope, rope = dev(ref[keys[0]]), dev(ref[keys[1]])
    r = {"cache_dtypes": [str(nope.dtype), str(rope.dtype)],
         "cache_shapes": [list(nope.shape), list(rope.shape)]}
    out = cf(nope, rope, ql, qr, kvc, kpe).cpu().view(torch.int16).numpy().view(np.uint16)
    args = (nope, rope, ql, qr, kvc, kpe)
    r["us"] = round(att._median_time(lambda: cf(*args), sync, a.iters, a.repeats)[0], 1)
    from pathlib import Path

    from sglang_tpu.profiler import start_tpu_profile, stop_tpu_profile

    d = f"{a.xprof_dir}/diag_{a.variant}_{a.case}"
    sync()
    trace = start_tpu_profile(output_dir=Path(d), with_cpu=False)
    try:
        for _ in range(a.iters):
            cf(*args)
        sync()
    finally:
        stop_tpu_profile(trace)
    r.update(_summ(d, a.iters))
    return r, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("jax_tc", "jax_sc", "torch_tc", "torch_sc"),
                   required=True)
    p.add_argument("--case", default="pr2198_extend_c1")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--npz-dir", required=True)
    p.add_argument("--xprof-dir", required=True)
    a = p.parse_args()
    c = att.case(a.case)
    ref = np.load(os.path.join(a.npz_dir, f"base_{a.case}.npz"))
    side, layout = a.variant.split("_")
    row = {"case": a.case, "variant": a.variant}
    try:
        r, out = (run_jax if side == "jax" else run_torch)(a, c, ref, layout)
        row.update(r)
        row["out_sha"] = __import__("hashlib").sha256(np.ascontiguousarray(out)).hexdigest()[:16]
    except Exception as e:  # report, do not hide
        import traceback

        row["error"] = repr(e)[:1500]
        traceback.print_exc()
    print("DIAG " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
