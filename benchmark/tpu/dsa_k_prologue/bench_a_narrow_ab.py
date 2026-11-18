"""Test only: Codex fixed A vs fixed A + narrow_output, same round.

Fixed A (sub3_20261118/REPORT.md): native split KV, phase_stride 5, compact RoPE,
rows_per_dma 32, atoms_per_batch 32, fold 1, chunk 128 / batch 16. Timed step as in
bench_fair_shared_ab.py: FP8 KV write (caches donated) + split-Q concat + all 16
attention chunks + output 512; the full tuple is synced. ``narrow`` makes the
kernel write the 512-wide output itself, so no trailing [..., :512] slice.
Real top-k (layer 0 / 10 / 40) plus synthetic; 10 rounds x 30 calls, order
alternated per round. Prints ``NAB`` lines.
"""

import glob
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_fp8_attention_ab as att  # noqa: E402
from check_physical_topk import to_physical  # noqa: E402

PAGE = att.FP = 64
CHUNK, BATCH, TOPK = 128, 16, 2048
ROUNDS, ITERS = 10, 30


def main():
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "/old/fj8"
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp
    from jax.experimental.layout import Format, Layout

    from sglang_tpu.attention.pallas.mla_sparse import kv_cache_utils as kcu
    from sglang_tpu.attention.pallas.mla_sparse.kernel import sparse_ragged_paged_attention

    c = att.case("pr2198_extend_c1", seed=0)
    sh = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    fmt = Format(Layout(major_to_minor=(0, 1, 2)), sh)
    put = lambda x: jax.device_put(x, sh)
    lay = kcu.KVCacheLayout.SPARSECORE
    ns = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.NOPE, lay, c["P"], PAGE, 512)
    rs = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.ROPE, lay, c["P"], PAGE, 64)
    q8 = lambda x: jnp.clip(jnp.asarray(x).astype(jnp.float32), -448, 448).astype(
        jnp.float8_e4m3fn)
    n = c["P"] * PAGE
    fill = jax.jit(lambda no, ro, kv, pe: kcu.update_sparse_mla_kv_cache_jax(
        no, ro, kv, pe, jnp.array([n], jnp.int32), jnp.arange(c["P"], dtype=jnp.int32),
        jnp.array([0, n], jnp.int32), nope_spec=ns, rope_spec=rs))
    cache = fill(jnp.zeros(ns.shape, ns.jax_dtype), jnp.zeros(rs.shape, rs.jax_dtype),
                 q8(c["prefix_kv"]), q8(c["prefix_pe"]))
    initial = tuple(np.asarray(x) for x in cache)
    del cache
    kvc, kpe = q8(c["kv_c"]), q8(c["k_pe"])
    kvl, pi, cuq = (put(c[k]) for k in ("kv", "pi", "cu_q"))
    dist = put(np.array([0, 0, 1], np.int32))
    qin = (jax.device_put(c["q_latent"], fmt), jax.device_put(c["q_rope"], fmt))
    t = c["T"]

    def make(narrow, rows):
        def step(no, ro, qs, kv, pe, ix):
            no, ro = kcu.update_sparse_mla_kv_cache(
                no, ro, kv, pe, kvl, pi, cuq, nope_spec=ns, rope_spec=rs)
            q = jnp.concatenate(qs, axis=-1)
            out = sparse_ragged_paged_attention(
                q, no, ro, ix, None, jnp.array([0, t], jnp.int32), dist,
                sm_scale=float(c["sm_scale"]), k_scale=1.0, cache_layout="sparsecore",
                gather_and_attention_chunk_size=CHUNK, attention_kernel_batch_size=BATCH,
                sparsecore_atoms_per_batch=32, sparsecore_gather_wait_mode="overlap_rope",
                sparsecore_phase_stride=5, compact_rope_output=True,
                sparsecore_rows_per_dma=32, sparsecore_batch_fold=1,
                narrow_output=narrow)
            return (out if narrow else out[..., :512]), no, ro

        jf = jax.jit(step, donate_argnums=(0, 1),
                     in_shardings=(sh, sh, (fmt, fmt), sh, sh, sh))
        st = {"cache": tuple(put(x) for x in initial), "last": None}
        hlo = jf.lower(*st["cache"], qin, kvc, kpe, rows).compile().as_text()

        def call():
            st["last"] = jf(*st["cache"], qin, kvc, kpe, rows)
            st["cache"] = st["last"][1:]

        call()
        jax.block_until_ready(st["last"])
        out = np.asarray(st["last"][0]).view(np.uint16)
        caches = [np.asarray(x) for x in st["cache"]]
        slices = sum(1 for ln in hlo.splitlines()
                     if "bf16[2048,64,512]" in ln and " slice(" in ln)
        return dict(call=call, sync=lambda: jax.block_until_ready(st["last"]),
                    out=out, caches=caches, slices=slices)

    paths = {os.path.basename(f)[:-4]: f
             for f in glob.glob(f"{raw_dir}/**/raw-*.npy", recursive=True)}
    sets = [("synthetic", np.asarray(c["topk"]).astype(np.int32))]
    for name in ("raw-0096-layer0", "raw-0097-layer10", "raw-0099-layer40"):
        if name in paths:
            sets.append((name, np.load(paths[name]).astype(np.int32)))
    for name, lg in sets:
        r = {"sample": name}
        try:
            rows = put(to_physical(lg, c["pi"], c["ppr"], c["cu_q"], page=PAGE))
            v = {"A": make(False, rows), "A_narrow": make(True, rows)}
            r["out_exact"] = bool(np.array_equal(v["A"]["out"], v["A_narrow"]["out"]))
            r["cache_exact"] = all(np.array_equal(x, y) for x, y in
                                   zip(v["A"]["caches"], v["A_narrow"]["caches"]))
            r["hlo_out_slices"] = {k: x["slices"] for k, x in v.items()}
            for f in v.values():
                for _ in range(3):
                    f["call"]()
                f["sync"]()
            keys, times = list(v), {k: [] for k in v}
            for rep in range(ROUNDS):
                for k in keys[rep % 2:] + keys[:rep % 2]:
                    f = v[k]
                    f["sync"]()
                    t0 = time.perf_counter()
                    for _ in range(ITERS):
                        f["call"]()
                    f["sync"]()
                    times[k].append((time.perf_counter() - t0) / ITERS * 1e6)
            med = {k: round(statistics.median(x), 1) for k, x in times.items()}
            r.update(median_us=med,
                     range_us={k: [round(min(x), 1), round(max(x), 1)]
                               for k, x in times.items()},
                     saved_us=round(med["A"] - med["A_narrow"], 1))
            del v
        except Exception as e:  # report, do not hide
            import traceback

            traceback.print_exc()
            r["fail"] = repr(e)[:500]
        print("NAB " + json.dumps(r), flush=True)
    print("NAB_DONE", flush=True)


if __name__ == "__main__":
    main()
