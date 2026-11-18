"""Test only: do duplicate rows in the real top-k slow the SparseCore gather?

Earlier runs put extend_c1 at ~3.41-3.49 ms with synthetic random top-k and
~3.70-3.98 ms with the real GLM-5.2 dump (exp-fj8ahrq5pz), in different runs.
Adjacent real queries share ~75% of their rows, and the gather microbench
(exp-h1pj43wtu2) was 8% slower with partly repeated indices. This A/B, one
process, same cache, measures every index order on two calls:

* ``gather``: ``dsa_gather_native_sc`` alone on chunk 0 (128 q x 2048 rows);
* ``attn``: the full ``sparse_ragged_paged_attention`` (SparseCore layout,
  physical slots, chunk 128 / batch 16 / atoms 16), as #2246 runs extend.

Index sets: ``random`` (synthetic, the old baseline), and per dump file
``raw`` (indexer order), ``sorted`` (each query ascending), ``shuffled``
(each query randomly permuted), ``rotated`` (sorted, query q rolled by 16*q so
neighbours read the same rows at different positions). Softmax does not depend
on row order; outputs are compared with ``raw`` (max_abs; not bit-exact since
the summation order changes). Prints ``DUP`` lines.
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
    files = (sys.argv[2] if len(sys.argv) > 2 else
             "raw-0096-layer0,raw-0097-layer10,raw-0099-layer40").split(",")
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

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

    attn = jax.jit(lambda q_, rows: sparse_ragged_paged_attention(
        q_, nope, rope, rows, None, jnp.array([0, T], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32), sm_scale=sm, k_scale=1.0,
        cache_layout="sparsecore", gather_and_attention_chunk_size=CHUNK,
        attention_kernel_batch_size=16, sparsecore_atoms_per_batch=16)[..., :512])
    gather = jax.jit(lambda ix: nsc.dsa_gather_native_sc(
        nope, rope, ix, out_size=CHUNK * TOPK, atoms_per_batch=16))

    def timed(fn, *args):
        st = {}

        def call():
            st["o"] = fn(*args)
            return st["o"]

        med, runs = att._median_time(call, lambda: jax.block_until_ready(st["o"]), 20, 5)
        return round(med, 1), [round(min(runs), 1), round(max(runs), 1)]

    def dup_stats(rows):
        # rows that some other query in the same 64-occurrence-aligned window also reads
        blk = rows[:CHUNK].reshape(-1)
        u, cnt = np.unique(blk, return_counts=True)
        return dict(distinct_frac=round(u.size / blk.size, 4), max_repeat=int(cnt.max()))

    rng = np.random.default_rng(7)
    sets = [("random", "-", np.asarray(c["topk"]).astype(np.int32))]
    paths = {os.path.basename(f)[:-4]: f
             for f in glob.glob(f"{raw_dir}/**/raw-*.npy", recursive=True)}
    for name in files:
        raw = np.load(paths[name]).astype(np.int32)
        srt = np.sort(raw, axis=1)
        sets += [(name, "raw", raw), (name, "sorted", srt),
                 (name, "shuffled", np.stack([rng.permutation(x) for x in raw])),
                 (name, "rotated", np.stack([np.roll(x, 16 * (i % CHUNK))
                                             for i, x in enumerate(srt)]))]
    refs = {}
    for name, order, lg in sets:
        r = {"file": name, "order": order}
        try:
            assert lg.shape == (T, TOPK), lg.shape
            pr = jnp.asarray(phys(lg))
            r.update(dup_stats(lg))
            r["gather_us"], r["gather_range"] = timed(gather, pr[:CHUNK])
            out = np.asarray(attn(q, pr)).astype(np.float32)
            if order == "raw":
                refs[name] = out
            if name in refs and order != "raw":
                r["max_abs_vs_raw"] = float(np.abs(out - refs[name]).max())
            r["finite"] = bool(np.isfinite(out).all())
            r["attn_us"], r["attn_range"] = timed(attn, q, pr)
        except Exception as e:  # report, do not hide
            import traceback

            traceback.print_exc()
            r["fail"] = repr(e)[:400]
        print("DUP " + json.dumps(r), flush=True)
    print("DUP_DONE", flush=True)


if __name__ == "__main__":
    main()
