"""Test only: where the SparseCore attention step's time goes (oracle + ablations).

JAX on TPU, #2246 head defaults for extend 2048 tok / 49K prefix / top-k 2048 /
page 64 / FP8 KV (packed RoPE cache, fixed-A gather, narrow output, split q),
16 chunks of 128 queries, the cache filled once (no KV write in the timed step).

Variants, each its own jit, all on the same cache and physical top-k:

* ``online``: SC gather + TC attention per chunk, as the torch op runs it.
* ``pre``: the same attention fed pre-gathered tiles (Codex's oracle): the
  gather runs once outside the timed call, so this is attention alone.
* ``pre+<ablation>``: ``pre`` with parts of the kernel body removed
  (``_attention_kernel(ablate=...)``): ``no_pv``, ``no_qk``, ``no_exp``,
  ``no_softmax`` (mask/max/exp/sum), ``no_kv_mask``, ``pv_bf16`` (bf16 p into
  PV instead of f32), and ``no_qk+no_softmax+no_pv`` (DMA + KV unpack only).
  Ablated outputs are wrong by design; only their time is read.
* ``gather``: the SC gather alone (16 chunks).

Differences between rows are not exact self-times (the kernel overlaps MXU,
VPU and DMA work), they bound how much each part can matter. ``online`` and
``pre`` outputs are compared bit for bit. Prints ``ABL`` lines.
"""

import argparse
import glob
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "tpu")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_fp8_attention_ab as att  # noqa: E402
from check_physical_topk import to_physical  # noqa: E402

FP = att.FP = 64
CASE = "pr2198_extend_c1"
CHUNK = 128
ABLATIONS = (
    ("no_pv",), ("no_qk",), ("no_exp",), ("no_softmax",), ("no_kv_mask",),
    ("pv_bf16",), ("no_qk", "no_softmax", "no_pv"),
)


def emit(**kw):
    print("ABL " + json.dumps(kw), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="synthetic,raw-0096-layer0")
    p.add_argument("--raw-dir", default="/old/fj8")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--xprof-dir", default="")
    a = p.parse_args()

    import jax
    import jax.numpy as jnp

    from sglang_tpu.attention.pallas.mla_sparse import kv_cache_utils as kcu
    from sglang_tpu.attention.pallas.mla_sparse import native_sc_gather as nsc
    from sglang_tpu.attention.pallas.mla_sparse.kernel import (
        SPARSECORE_CACHE_LAYOUT,
        prepare_gather_indices,
        sparse_attention_chunk,
    )
    from sglang_tpu.attention.pallas.mla_sparse_fused import (
        _sparsecore_atoms,
        _sparsecore_gather_kwargs,
    )

    c = att.case(CASE)
    P, n, T, K = c["P"], c["P"] * FP, c["T"], c["topk"].shape[1]
    lay = kcu.KVCacheLayout.SPARSECORE
    ns = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.NOPE, lay, P, FP, 512)
    rs = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.ROPE, lay, P, FP, 64)
    f8 = jnp.float8_e4m3fn
    fmax = float(jnp.finfo(f8).max)
    q8 = lambda x: jnp.clip(x.astype(jnp.float32), -fmax, fmax).astype(f8)  # noqa: E731
    nope, rope = jax.jit(lambda no, ro, kv_, pe_: kcu.update_sparse_mla_kv_cache_jax(
        no, ro, q8(kv_), q8(pe_), jnp.array([n], jnp.int32), jnp.arange(P, dtype=jnp.int32),
        jnp.array([0, n], jnp.int32), nope_spec=ns, rope_spec=rs))(
        jnp.zeros(ns.shape, ns.jax_dtype), jnp.zeros(rs.shape, rs.jax_dtype),
        jnp.asarray(c["prefix_kv"]), jnp.asarray(c["prefix_pe"]))
    ql = jnp.asarray(c["q_latent"])
    qr = jnp.asarray(c["q_rope"])
    sm = float(c["sm_scale"])
    atoms = _sparsecore_atoms(T, CHUNK, K)
    gkw = _sparsecore_gather_kwargs(nope, rope, K, atoms)
    rope_rows = K // (nsc.ROPE_PACKING * (2 if gkw.get("compact_rope_output") else 1))
    emit(stage="setup", T=T, topk=K, pages=P, atoms=atoms, gather_kwargs=gkw,
         nope=list(nope.shape), rope=list(rope.shape))
    # 2 matmuls (QK over 576 live dims, PV over 512) per (query, head, row).
    flops = 2 * T * ql.shape[1] * K * (576 + 512)

    def gather(phys):
        tiles = []
        for i in range(T // CHUNK):
            gn, gr = nsc.dsa_gather_native_sc(
                nope, rope, phys[i * CHUNK:(i + 1) * CHUNK], out_size=CHUNK * K,
                atoms_per_batch=atoms, phase_stride=gkw.get("sparsecore_phase_stride", 0),
                compact_rope_output=gkw.get("compact_rope_output", False),
                rows_per_dma=gkw["sparsecore_rows_per_dma"])
            tiles.append((gn.reshape(CHUNK, K, 128), gr.reshape(CHUNK, rope_rows, 128)))
        return tuple(tiles)

    def attend(nope_, rope_, ql_, qr_, phys, kv_lens, tiles=None, ablate=()):
        o = None
        for i in range(T // CHUNK):
            o = sparse_attention_chunk(
                ql_, nope_, rope_, kv_lens, phys, None, start_pos=i * CHUNK,
                end_pos=(i + 1) * CHUNK, token_end=jnp.int32(T), sm_scale=sm,
                k_scale=1.0, cache_layout=SPARSECORE_CACHE_LAYOUT,
                sparsecore_atoms_per_batch=atoms, topk_override=K, narrow_output=True,
                o=o, q_rope=qr_, ablate=ablate,
                gathered_group=None if tiles is None else tiles[i], **gkw)
        return o

    def timed(f, *args):
        def call():
            return jax.block_until_ready(f(*args))
        med, runs = att._median_time(call, lambda: None, a.iters, a.repeats)
        return round(med, 1), [round(min(runs), 1), round(max(runs), 1)]

    def profile(f, args, tag):
        if not a.xprof_dir:
            return {}
        from xplane_summary import breakdown

        d = os.path.join(a.xprof_dir, tag)
        jax.block_until_ready(f(*args))
        jax.profiler.start_trace(d)
        try:
            for _ in range(10):
                out = f(*args)
            jax.block_until_ready(out)
        finally:
            jax.profiler.stop_trace()
        try:
            b = breakdown(d, 10, [("tc_attention", ("MLA-sparsecore",)),
                                  ("sc_gather", ("native_sc_gather",))])
            return {"module_us": b.get("module_span_us"), "ops_busy_us": b.get("ops_busy_us"),
                    "cats": b.get("categories"),
                    "lines": dict(list(b.get("lines", {}).items())[:8])}
        except Exception as e:  # report, do not hide
            return {"prof_error": repr(e)[:400]}

    paths = {os.path.basename(f)[:-4]: f
             for f in glob.glob(f"{a.raw_dir}/**/raw-*.npy", recursive=True)}
    for name in a.samples.split(","):
        try:
            lg = (np.asarray(c["topk"]) if name == "synthetic"
                  else np.load(paths[name])).astype(np.int32)
            phys_np = to_physical(lg, c["pi"], c["ppr"], c["cu_q"], page=FP)
            kv_lens, phys = prepare_gather_indices(
                jnp.asarray(phys_np), None, jnp.asarray(c["cu_q"]), page_size=FP,
                total_num_pages=P)
            args = (nope, rope, ql, qr, phys, kv_lens)
            online = jax.jit(attend)
            tiles = jax.block_until_ready(jax.jit(gather)(phys))
            ref = np.asarray(online(*args).view(jnp.uint16))

            us, rng = timed(online, *args)
            emit(sample=name, variant="online", us=us, us_range=rng,
                 tflops=round(flops / us / 1e6, 1), **profile(online, args, f"{name}_online"))
            g = jax.jit(gather)
            us, rng = timed(g, phys)
            emit(sample=name, variant="gather", us=us, us_range=rng,
                 **profile(g, (phys,), f"{name}_gather"))
            for ab in ((),) + ABLATIONS:
                tag = "pre" + "".join("+" + x for x in ab)
                f = jax.jit(lambda *xs, _ab=ab: attend(*xs[:6], tiles=xs[6], ablate=_ab))
                r = dict(sample=name, variant=tag)
                try:
                    out = np.asarray(f(*args, tiles).view(jnp.uint16))
                    if not ab:
                        r["exact_vs_online"] = bool(np.array_equal(out, ref))
                    r["us"], r["us_range"] = timed(f, *args, tiles)
                    if not ab:
                        r["tflops"] = round(flops / r["us"] / 1e6, 1)
                        r.update(profile(f, (*args, tiles), f"{name}_pre"))
                except Exception as e:  # report, do not hide; next ablation still runs
                    r["fail"] = repr(e)[:600]
                emit(**r)
            del tiles
        except Exception as e:  # report, do not hide
            import traceback

            traceback.print_exc()
            emit(sample=name, fail=repr(e)[:800])
    print("ABL_DONE", flush=True)


if __name__ == "__main__":
    main()
