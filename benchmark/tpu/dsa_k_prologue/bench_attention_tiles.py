"""Tile sweep for the Ported-S FP8 attention (S5-S6) at the PR 2198 shapes, page 64.

Runs ``dsa_fp8_attention.fp8_attention_step(topk_is_physical=True, tiles=...)`` in
JAX on TPU (the torch op runs the same JAX function; earlier runs matched it byte
for byte and within ~1% in time). For each case, every candidate
``(bq_sparse, bq, b_topk)`` is compiled, timed and profiled; the caches are donated
so no cache copy is timed. Reported per tile: device module time, the TensorCore
attention and SparseCore gather time, and the error vs f64 attention on unquantized
K/V. ``plan`` marks the tiles ``DsaConfig.resolve`` picks today.
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
CASES = ("pr2198_extend_c1", "pr2198_verify_b4", "pr2198_decode_b4")


def candidates(T):
    if T >= 128:
        return [(s, q, k) for s in (128, 256, 512) for q in (8, 16) for k in (256, 512, 1024)]
    if T == 24:
        return [(s, q, k) for s, q in ((24, 8), (24, 12), (12, 12), (12, 6), (8, 8), (6, 6))
                for k in (256, 512, 1024)]
    return [(s, q, k) for s, q in ((4, 4), (4, 2), (4, 1), (2, 2), (2, 1)) for k in (256, 512, 1024)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=list(CASES))
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--xprof-dir", required=True)
    a = p.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    from xplane_summary import summarize
    from sglang_tpu.attention.pallas import dsa_fp8_attention as port
    from sglang_tpu.attention.pallas.source_dsa.configs.dsa_config import DsaConfig
    from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.fp8_kv_cache import SplitFP8KVCache
    from sglang_tpu.attention.pallas.source_dsa.layers.attention.dsa_sparse_backend_helpers import (
        _scatter_fused_kv_paged,
    )

    for name in a.cases:
        c = att.case(name)
        P, n, T = c["P"], c["P"] * FP, c["T"]
        prefill = c["mode"] == "prefill"
        kv, pe = jnp.asarray(c["prefix_kv"]), jnp.asarray(c["prefix_pe"])
        empty = SplitFP8KVCache(jnp.zeros((P, FP, 4, 128), jnp.uint8),
                                jnp.zeros((P, FP // 4, 4, 128), jnp.uint8))
        filled = jax.jit(lambda cc, kv_, pe_: _scatter_fused_kv_paged(
            cc, kv_, pe_, jnp.array([n], jnp.int32), jnp.arange(P, dtype=jnp.int32),
            jnp.array([0, n], jnp.int32), jnp.array([0, n], jnp.int32),
            kv_lora_rank=att.LAT, fp8_dtype=jnp.float8_e4m3fn))(empty, kv, pe)
        lat0, rope0 = np.asarray(filled.latent), np.asarray(filled.rope)
        rows_bf = att.bf16_rows(c)
        slots = jnp.asarray(to_physical(c["topk"], c["pi"], c["ppr"], c["cu_q"], page=FP))
        ins = tuple(jnp.asarray(c[k]) for k in ("q_latent", "q_rope", "kv_c", "k_pe"))
        meta = (jnp.asarray(c["kv"]), jnp.asarray(c["pi"]), jnp.asarray(c["cu_q"]))
        sm = jnp.float32(c["sm_scale"])
        plan = DsaConfig(precision="fp8-qk-pv").resolve(
            T, eligible=True, is_prefill=prefill, fp8_kv=True).tiles
        print("TILEPLAN " + json.dumps({"case": name, "T": T, "plan": plan}), flush=True)

        for tiles in candidates(T):
            row = {"case": name, "tiles": tiles, "plan": tuple(tiles) == tuple(plan)}
            try:
                def step(lat, rope, ql, qr, kvc, kpe, kvl, pi, cuq, sm_, _t=tiles):
                    return port.fp8_attention_step(
                        lat, rope, ql, qr, kvc, kpe, slots, kvl, pi, cuq, sm_, allow_fp8_q=True,
                        is_prefill=prefill, topk_is_physical=True, tiles=_t)

                jf = jax.jit(step, donate_argnums=(0, 1))
                st = {"c": (jnp.asarray(lat0), jnp.asarray(rope0))}

                def call():
                    out, la, ro = jf(*st["c"], *ins, *meta, sm)
                    st["c"] = (la, ro)
                    return out

                out = np.asarray(call()).astype(np.float64)
                row["error"] = att.error_vs_bf16(c, out, rows_bf)
                sync = lambda: jax.block_until_ready(st["c"][0])
                row["us"] = round(att._median_time(call, sync, a.iters, a.repeats)[0], 1)
                d = f"{a.xprof_dir}/tiles_{name}_{'_'.join(map(str, tiles))}"
                sync()
                jax.profiler.start_trace(d)
                for _ in range(a.iters):
                    call()
                sync()
                jax.profiler.stop_trace()
                x = summarize(d, a.iters, top=200)
                ops = x.get("ops", [])
                row["module_us"] = round(sum(u for _, u in x.get("modules", [])), 1)
                row["tc_us"] = round(sum(u for nm, u in ops if "dsa_tensor_core_attention" in nm), 1)
                row["gather_us"] = round(sum(u for nm, u in ops if "sparse_core_gather" in nm), 1)
            except Exception as e:  # an invalid or over-VMEM tile; report and go on
                row["fail"] = repr(e)[:300]
            print("TILE " + json.dumps(row), flush=True)
    print("TILES_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
