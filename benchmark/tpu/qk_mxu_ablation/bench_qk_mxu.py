"""Q x K ablation on the TPU7x MXU: four ways to multiply BF16 queries by FP8 keys.

Only the score matmul ``S = Q @ K`` of sparse attention is timed (no softmax, no PV,
no gather). Queries are BF16 ``[M, D]`` (M = heads x query tile), keys are stored FP8
E4M3 ``[D, N]`` (N = keys per step), D = 512 (the NoPE width). ``G`` independent
groups (one per grid step) make the kernel long enough to time.

* ``fp8``      A. Q and K both FP8 before the kernel; FP8 x FP8 dot (upper bound).
* ``k_dequant`` B. Source bf16 path: K widened to BF16 on the VPU, BF16 x BF16 dot.
* ``q_quant``  C. Source fp8-qk path: Q quantized to FP8 in the kernel (per-row amax
  scale), FP8 x FP8 dot, then rescaled.
* ``w8a16_dot`` D1. Target style, written as a mixed BF16 x FP8 ``dot_general``; the
  compiler picks the lowering.
* ``w8a16_mxu`` D2. Target style, explicit MXU ops as ``pallas-samples``
  ``gemm_mxu.w8a16_kernel``: raw FP8 K pushed into MSRA, widened on the fused GMR
  load, BF16 multiply into ACC, FP32 pop.
* ``bf16_mxu``  B2. Same explicit MXU kernel with K already BF16 (``gemm_mxu``
  staging): isolates the cost of the VPU widen in B from the MXU schedule.

Every variant uses the same grid and blocks (``bm`` x ``bn``, full D per block),
reads the same bytes per operand except where the variant itself changes them,
and writes BF16 scores. Reported: device time per call (xprof), TFLOP/s, wall time,
and max / median relative error vs f64 on the stored (dequantized) operands.
Inputs are seeded ``jax.random.normal`` cast to each dtype, as in pallas-samples.
"""

import argparse
import functools
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dsa_k_prologue"))

VARIANTS = ("fp8", "k_dequant", "q_quant", "w8a16_dot", "w8a16_mxu", "bf16_mxu")
FP8_MAX = 448.0


def make_kernels(jax, jnp, pl, pltpu):
    f8 = jnp.float8_e4m3fn
    dot = functools.partial(jax.lax.dot, preferred_element_type=jnp.float32)

    def plain(mode):
        def kernel(q_ref, k_ref, o_ref):
            q, k = q_ref[...], k_ref[...]
            if mode == "fp8":
                s = dot(q, k)
            elif mode == "k_dequant":
                s = dot(q, k.astype(jnp.bfloat16))
            elif mode == "q_quant":
                amax = jnp.max(jnp.abs(q.astype(jnp.float32)), axis=1, keepdims=True)
                sc = jnp.maximum(amax, 1e-12) / FP8_MAX
                s = dot((q.astype(jnp.float32) / sc).astype(f8), k) * sc
            elif mode == "w8a16_dot":
                s = dot(q, k)
            else:
                raise ValueError(mode)
            o_ref[...] = s.astype(o_ref.dtype)
        return kernel

    def mxu(lhs_ref, rhs_ref, out_ref):
        # pallas-samples gemm_mxu_kernel (staging / w8a16), unchanged schedule.
        bm, k_size = lhs_ref.shape
        bn = rhs_ref.shape[1]
        tile_m, tile_k, tile_n = 64, 256, 256
        accs = [[jax.empty_ref(jax.ShapeDtypeStruct((tile_m, tile_n), jnp.float32),
                               memory_space=pltpu.ACC(m))
                 for _ in range(bm // tile_m)] for m in range(2)]

        def stage_rhs(k, n):
            for m in range(2):
                pltpu.matmul_push_rhs(rhs_ref[pl.ds(k, tile_k), pl.ds(n + m * tile_n, tile_n)],
                                      staging_register=0, mxu_index=m)

        for n in range(0, bn, 2 * tile_n):
            stage_rhs(0, n)
            for k in range(0, k_size, tile_k):
                for r in range(0, bm, tile_m):
                    for m in range(2):
                        pltpu.matmul_acc_lhs(accs[m][r // tile_m],
                                             lhs_ref[pl.ds(r, tile_m), pl.ds(k, tile_k)],
                                             load_staged_rhs=0 if r == 0 else None)
                    if r == 0 and k + tile_k < k_size:
                        stage_rhs(k + tile_k, n)
            for r in range(0, bm, tile_m):
                for m in range(2):
                    out_ref[pl.ds(r, tile_m), pl.ds(n + m * tile_n, tile_n)] = (
                        pltpu.matmul_pop(accs[m][r // tile_m]).astype(out_ref.dtype))

    return {"fp8": plain("fp8"), "k_dequant": plain("k_dequant"),
            "q_quant": plain("q_quant"), "w8a16_dot": plain("w8a16_dot"),
            "w8a16_mxu": mxu, "bf16_mxu": mxu}


def build_op(jax, jnp, pl, pltpu, kernel, *, G, M, N, D, bm, bn, name):
    return jax.jit(lambda q, k: pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((G, M, N), jnp.bfloat16),
        grid=(G, M // bm, N // bn),
        in_specs=[pl.BlockSpec((None, bm, D), lambda g, i, j: (g, i, 0)),
                  pl.BlockSpec((None, D, bn), lambda g, i, j: (g, 0, j))],
        out_specs=pl.BlockSpec((None, bm, bn), lambda g, i, j: (g, i, j)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel")),
        name=f"qk_{name}",
    )(q, k))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--groups", type=int, default=32)
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--xprof-dir", required=True)
    a = p.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp
    import ml_dtypes
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from xplane_summary import summarize

    info = pltpu.get_tpu_info()
    print("ENV " + json.dumps({"jax": jax.__version__, "device": str(jax.devices()[0]),
                               "chip": str(getattr(info, "chip_version", "?")),
                               "mxu_api": hasattr(pltpu, "matmul_push_rhs")}), flush=True)
    kernels = make_kernels(jax, jnp, pl, pltpu)
    f8 = jnp.float8_e4m3fn
    D = a.d
    # (M, N, bm, bn): DSA-like blocks (M = 64 heads x 2 / 4 / 8 / 16 queries), then a
    # large block for raw throughput.
    shapes = [(m, 2048, m, 512) for m in (128, 256, 512, 1024)] + [(1024, 4096, 1024, 1024)]

    for M, N, bm, bn in shapes:
        G = a.groups
        kq, kk = jax.random.split(jax.random.key(0))
        q_bf = jax.random.normal(kq, (G, M, D), jnp.float32).astype(jnp.bfloat16)
        k_f8 = jax.random.normal(kk, (G, D, N), jnp.float32).astype(f8)
        q_f8 = q_bf.astype(f8)
        k_bf = k_f8.astype(jnp.bfloat16)  # same values as k_f8, exactly
        flops = 2.0 * G * M * N * D
        # f64 reference on the stored operands (BF16 Q, FP8 K); sampled groups/rows.
        qn = np.asarray(q_bf.astype(jnp.float32), np.float64)
        kn = np.asarray(k_f8.astype(jnp.float32), np.float64)
        q8n = np.asarray(q_f8.astype(jnp.float32), np.float64)
        sample_g = (0, G // 2, G - 1)
        for var in a.variants:
            row = {"M": M, "N": N, "D": D, "G": G, "bm": bm, "bn": bn, "variant": var}
            try:
                q_in = q_f8 if var == "fp8" else q_bf
                k_in = k_bf if var == "bf16_mxu" else k_f8
                op = build_op(jax, jnp, pl, pltpu, kernels[var], G=G, M=M, N=N, D=D,
                              bm=bm, bn=bn, name=var)
                out = op(q_in, k_in)
                out.block_until_ready()
                o = np.asarray(out.astype(jnp.float32), np.float64)
                errs = []
                for g in sample_g:
                    ref = (q8n[g] if var == "fp8" else qn[g]) @ kn[g]
                    errs.append(np.abs(o[g] - ref).max() / np.abs(ref).max())
                row["max_rel"] = float(max(errs))
                runs = []
                for _ in range(a.repeats):
                    t0 = time.perf_counter()
                    for _ in range(a.iters):
                        out = op(q_in, k_in)
                    out.block_until_ready()
                    runs.append((time.perf_counter() - t0) / a.iters * 1e6)
                row["wall_us"] = round(statistics.median(runs), 1)
                d = f"{a.xprof_dir}/qk_{var}_{M}x{N}"
                jax.profiler.start_trace(d)
                for _ in range(a.iters):
                    out = op(q_in, k_in)
                out.block_until_ready()
                jax.profiler.stop_trace()
                x = summarize(d, a.iters, top=10)
                kern = sum(u for n, u in x.get("ops", []) if f"qk_{var}" in n)
                row["kernel_us"] = round(kern, 2)
                row["module_us"] = round(sum(u for _, u in x.get("modules", [])), 2)
                t = kern or row["module_us"]
                row["tflops"] = round(flops / (t * 1e-6) / 1e12, 1) if t else None
                row["top_ops"] = [(n[:50], round(u, 2)) for n, u in x.get("ops", [])[:3]]
            except Exception as e:  # unsupported lowering or VMEM; report and go on
                row["fail"] = repr(e)[:600]
            print("QK " + json.dumps(row), flush=True)
    print("QK_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
