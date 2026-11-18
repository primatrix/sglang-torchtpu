"""Test only: how fast is #2246's extend attention with the gather taken out?

Same question as the Codex "fixed A" oracle (online 3.23-3.32 ms vs
pre-gathered ~2.76 ms), on #2246's configuration: SparseCore layout, physical
slots, chunk 128 / batch 16 / atoms 16 / overlap_rope, extend_c1 (2048 q,
49K prefix). One process, same cache, interleaved:

* ``online``: ``sparse_ragged_paged_attention`` as #2246 runs it (SC gather per
  chunk, then the TensorCore attention kernel);
* ``pregathered``: the same call with ``pregathered=`` (test-only kernel hook):
  every chunk's NoPE / RoPE rows were gathered before the timed call with the
  same ``dsa_gather_native_sc``, so the timed call runs only the attention
  kernels on ready buffers. Preparation time is excluded.

The difference is what removing the gather saves on the whole step, overlap
included; it is an upper bound, not a gather kernel time. Outputs must be
bit-identical. Top-k: synthetic random and the real GLM-5.2 dump (layers
0 / 10 / 40). Also an xprof of each variant (module time, top ops). Measured
with ``jax.jit``, not through the torch op. Prints ``PGA`` lines.
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_fp8_attention_ab as att  # noqa: E402

FP = att.FP = 64
CASE = "pr2198_extend_c1"
CHUNK, TOPK = 128, 2048


def main():
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "/old/fj8"
    xp = sys.argv[2] if len(sys.argv) > 2 else "/tmp/pga_xprof"
    files = "raw-0096-layer0,raw-0097-layer10,raw-0099-layer40".split(",")
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    from xplane_summary import summarize
    from sglang_tpu.attention.pallas.mla_sparse import kv_cache_utils as kcu
    from sglang_tpu.attention.pallas.mla_sparse import native_sc_gather as nsc
    from sglang_tpu.attention.pallas.mla_sparse.kernel import sparse_ragged_paged_attention

    f8 = jnp.float8_e4m3fn
    fmax = float(jnp.finfo(f8).max)
    q8 = lambda x: jnp.clip(x.astype(jnp.float32), -fmax, fmax).astype(f8)
    lay = kcu.KVCacheLayout.SPARSECORE
    c = att.case(CASE)
    P, T = c["P"], c["T"]
    n = P * FP
    ns = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.NOPE, lay, P, FP, 512)
    rs = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.ROPE, lay, P, FP, 64)
    fill = jax.jit(lambda nope, rope, kv_, pe_: kcu.update_sparse_mla_kv_cache_jax(
        nope, rope, kv_, pe_, jnp.array([n], jnp.int32), jnp.arange(P, dtype=jnp.int32),
        jnp.array([0, n], jnp.int32), nope_spec=ns, rope_spec=rs))
    nope, rope = fill(jnp.zeros(ns.shape, ns.jax_dtype), jnp.zeros(rs.shape, rs.jax_dtype),
                      q8(jnp.asarray(c["prefix_kv"])), q8(jnp.asarray(c["prefix_pe"])))
    q = jnp.concatenate([jnp.asarray(c["q_latent"]), jnp.asarray(c["q_rope"])], axis=-1)
    sm = float(c["sm_scale"])
    pi = np.asarray(c["pi"])
    phys = lambda lg: np.where(lg >= 0, pi[np.maximum(lg, 0) // FP] * FP
                               + np.maximum(lg, 0) % FP, -1).astype(np.int32)
    kw = dict(sm_scale=sm, k_scale=1.0, cache_layout="sparsecore",
              gather_and_attention_chunk_size=CHUNK, attention_kernel_batch_size=16,
              sparsecore_atoms_per_batch=16)
    cu, dist = jnp.array([0, T], jnp.int32), jnp.array([0, 0, 1], jnp.int32)

    online = jax.jit(lambda q_, rows: sparse_ragged_paged_attention(
        q_, nope, rope, rows, None, cu, dist, **kw)[..., :512])
    pre = jax.jit(lambda q_, rows, gn, gr: sparse_ragged_paged_attention(
        q_, nope, rope, rows, None, cu, dist, pregathered=(gn, gr), **kw)[..., :512])

    @jax.jit
    def prepare(rows):
        gn, gr = [], []
        for ci in range(T // CHUNK):
            a, b = nsc.dsa_gather_native_sc(nope, rope, rows[ci * CHUNK:(ci + 1) * CHUNK],
                                            out_size=CHUNK * TOPK, atoms_per_batch=16)
            gn.append(a.reshape(CHUNK, TOPK, 128))
            gr.append(b.reshape(CHUNK, TOPK // nsc.ROPE_PACKING, 128))
        return jnp.stack(gn), jnp.stack(gr)

    def timed(fn, *args):
        st = {}

        def call():
            st["o"] = fn(*args)
            return st["o"]

        med, runs = att._median_time(call, lambda: jax.block_until_ready(st["o"]), 20, 5)
        return round(med, 1), [round(min(runs), 1), round(max(runs), 1)]

    def prof(tag, fn, *args):
        d = f"{xp}/{tag}"
        jax.block_until_ready(fn(*args))
        jax.profiler.start_trace(d)
        for _ in range(5):
            o = fn(*args)
        jax.block_until_ready(o)
        jax.profiler.stop_trace()
        s = summarize(d, 5, top=6)
        return dict(module_us=round(sum(u for _, u in s.get("modules", [])), 1),
                    top_ops=[(nm[:44], round(u, 1)) for nm, u in s.get("ops", [])[:6]])

    sets = [("random", np.asarray(c["topk"]).astype(np.int32))]
    paths = {os.path.basename(f)[:-4]: f
             for f in glob.glob(f"{raw_dir}/**/raw-*.npy", recursive=True)}
    for name in files:
        if name in paths:
            sets.append((name, np.load(paths[name]).astype(np.int32)))
    for name, lg in sets:
        r = {"topk": name}
        try:
            rows = jnp.asarray(phys(lg))
            gn, gr = prepare(rows)
            o1 = np.asarray(online(q, rows))
            o2 = np.asarray(pre(q, rows, gn, gr))
            r["exact"] = bool(np.array_equal(o1.view(np.uint16), o2.view(np.uint16)))
            # Interleave: online, pre, online, pre.
            a1, ar1 = timed(online, q, rows)
            b1, br1 = timed(pre, q, rows, gn, gr)
            a2, ar2 = timed(online, q, rows)
            b2, br2 = timed(pre, q, rows, gn, gr)
            r.update(online_us=[a1, a2], online_range=ar1 + ar2,
                     pre_us=[b1, b2], pre_range=br1 + br2,
                     saved_us=round((a1 + a2 - b1 - b2) / 2, 1),
                     saved_frac=round(1 - (b1 + b2) / (a1 + a2), 3))
            r["online_prof"] = prof(f"{name}_online", online, q, rows)
            r["pre_prof"] = prof(f"{name}_pre", pre, q, rows, gn, gr)
            del gn, gr
        except Exception as e:  # report, do not hide
            import traceback

            traceback.print_exc()
            r["fail"] = repr(e)[:500]
        print("PGA " + json.dumps(r), flush=True)
    print("PGA_DONE", flush=True)


if __name__ == "__main__":
    main()
