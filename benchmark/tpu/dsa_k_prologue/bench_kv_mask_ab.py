"""Test only: does skipping the kv_len key mask on full blocks speed up attention?

The LLO of the SparseCore-layout attention kernel (exp-b2d2qoiwr4) charged ~20%
of its instructions to ``unpack_bkv``'s ``jnp.where(k_span < kv_len, bkv, 0)``,
which is the identity when every row of the block is live (extend: 2048 of
2048). ``kernel._KV_TAIL_MASK_MODE = "cond"`` (test-only, read at trace time)
wraps it in ``lax.cond(kv_len < bkv_sz, ...)``.

#2246 configuration (SparseCore layout, physical slots, chunk 128 / batch 16 /
atoms 16), extend_c1, ``jax.jit``. Each (mode, path) is AOT-compiled with the
caches cleared in between, so the mode reaches the kernel; the compiled texts
are compared to prove the two modes differ. Paths: ``online`` (SC gather +
attention) and ``pre`` (pre-gathered buffers, attention only). Top-k sets:
``random`` (all 2048 live), ``raw-0096-layer0`` (real dump), and ``tail``
(random, odd queries keep only 1000 rows: exercises the masked branch).
Outputs of ``cond`` must equal ``always`` bit for bit. Prints ``KMA`` lines.
"""

import glob
import hashlib
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
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    from sglang_tpu.attention.pallas.mla_sparse import kernel as kmod
    from sglang_tpu.attention.pallas.mla_sparse import kv_cache_utils as kcu
    from sglang_tpu.attention.pallas.mla_sparse import native_sc_gather as nsc

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
    spra = kmod.sparse_ragged_paged_attention

    def online_fn(q_, rows):
        return spra(q_, nope, rope, rows, None, cu, dist, **kw)[..., :512]

    def pre_fn(q_, rows, gn, gr):
        return spra(q_, nope, rope, rows, None, cu, dist, pregathered=(gn, gr), **kw)[..., :512]

    @jax.jit
    def prepare(rows):
        gn, gr = [], []
        for ci in range(T // CHUNK):
            a, b = nsc.dsa_gather_native_sc(nope, rope, rows[ci * CHUNK:(ci + 1) * CHUNK],
                                            out_size=CHUNK * TOPK, atoms_per_batch=16)
            gn.append(a.reshape(CHUNK, TOPK, 128))
            gr.append(b.reshape(CHUNK, TOPK // nsc.ROPE_PACKING, 128))
        return tuple(gn), tuple(gr)

    def timed(fn, *args):
        st = {}

        def call():
            st["o"] = fn(*args)
            return st["o"]

        med, runs = att._median_time(call, lambda: jax.block_until_ready(st["o"]), 20, 5)
        return round(med, 1), [round(min(runs), 1), round(max(runs), 1)]

    rng = np.random.default_rng(3)
    base = np.asarray(c["topk"]).astype(np.int32)
    tail = base.copy()
    tail[1::2, 1000:] = -1
    sets = [("random", base), ("tail", tail)]
    paths = {os.path.basename(f)[:-4]: f
             for f in glob.glob(f"{raw_dir}/**/raw-*.npy", recursive=True)}
    if "raw-0096-layer0" in paths:
        sets.append(("raw-0096-layer0", np.load(paths["raw-0096-layer0"]).astype(np.int32)))

    for name, lg in sets:
        r = {"topk": name}
        try:
            rows = jnp.asarray(phys(lg))
            gn, gr = prepare(rows)
            comp, digest = {}, {}
            for mode in ("always", "cond"):
                jax.clear_caches()
                kmod._KV_TAIL_MASK_MODE = mode
                comp[("online", mode)] = jax.jit(online_fn).lower(q, rows).compile()
                comp[("pre", mode)] = jax.jit(pre_fn).lower(q, rows, gn, gr).compile()
                digest[mode] = hashlib.sha1(
                    comp[("pre", mode)].as_text().encode()).hexdigest()[:12]
            kmod._KV_TAIL_MASK_MODE = "always"
            r["compiled_differ"] = digest["always"] != digest["cond"]
            args = {"online": (q, rows), "pre": (q, rows, gn, gr)}
            for path in ("online", "pre"):
                oa = np.asarray(comp[(path, "always")](*args[path]))
                oc = np.asarray(comp[(path, "cond")](*args[path]))
                r[f"{path}_exact"] = bool(np.array_equal(oa.view(np.uint16),
                                                         oc.view(np.uint16)))
                r[f"{path}_finite"] = bool(np.isfinite(oc.astype(np.float32)).all())
                # Interleave: always, cond, always, cond.
                a1, ar1 = timed(comp[(path, "always")], *args[path])
                b1, br1 = timed(comp[(path, "cond")], *args[path])
                a2, ar2 = timed(comp[(path, "always")], *args[path])
                b2, br2 = timed(comp[(path, "cond")], *args[path])
                r[path] = dict(always_us=[a1, a2], cond_us=[b1, b2],
                               always_range=ar1 + ar2, cond_range=br1 + br2,
                               saved_us=round((a1 + a2 - b1 - b2) / 2, 1),
                               saved_frac=round(1 - (b1 + b2) / (a1 + a2), 4))
            del gn, gr, comp
        except Exception as e:  # report, do not hide
            import traceback

            traceback.print_exc()
            r["fail"] = repr(e)[:500]
        print("KMA " + json.dumps(r), flush=True)
    print("KMA_DONE", flush=True)


if __name__ == "__main__":
    main()
