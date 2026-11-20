"""Test only (experiment B): Source TC kernel with a 2D latent tile vs the original.

extend_c1, physical slots, the Source fp8-qk-pv kernel on pre-gathered tiles
(the same ``src_pre`` as bench_src_vs_target.py). Two kernels:

* ``orig``: source_dsa fp8_tensor_core_attention, latent tile [bq, b_topk, 4, 128].
* ``tile2d``: fp8_attention_2d.py, the same kernel with the latent tile staged
  as [bq, b_topk * 4, 128] (one change), so Mosaic can pick (32,128) tiling.

Prints ``T2D`` lines: device module time per variant (profiler), bit-exact
check of tile2d vs orig, error vs f64 attention, then (compile only, LLO and
Mosaic dumps on) the latent memref tiling and per-source-line instruction
counts of the k loop for both kernels. Production code is untouched.
"""

import collections
import glob
import json
import os
import re
import sys

DUMP, MOSAIC = "/tmp/llo_dump", "/tmp/mosaic_dump"
for d in (DUMP, MOSAIC):
    os.makedirs(d, exist_ok=True)
os.environ.setdefault("JAX_PLATFORMS", "tpu")
os.environ["LIBTPU_INIT_ARGS"] = (
    os.environ.get("LIBTPU_INIT_ARGS", "")
    + f" --xla_jf_dump_to={DUMP} --xla_jf_dump_llo_text=true"
    " --xla_mosaic_enable_llo_source_annotations=true"
    f" --xla_mosaic_dump_to={MOSAIC}").strip()

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_fp8_attention_ab as att  # noqa: E402
from check_physical_topk import to_physical  # noqa: E402

FP = att.FP = 64
BUNDLE_RE = re.compile(r"\s*(0x[0-9a-f]+)\s*:\s*(>*)\s*\{(.*)\}\s*$")
TOK_RE = re.compile(r"(?<![%\w.$#])([a-z][a-z0-9_]*(?:\.[\w]+)*)")
LOC_RE = re.compile(r'"([^"]+\.py)":(\d+)')


def emit(**kw):
    print("T2D " + json.dumps(kw), flush=True)


def files(root):
    return {f for f in glob.glob(f"{root}/**/*", recursive=True) if os.path.isfile(f)}


def llo_lines(tag, new, name):
    fs = sorted(f for f in new if name in os.path.basename(f)
                and os.path.basename(f).endswith("final_bundles.txt")
                and "schedule-analysis" not in os.path.basename(f))
    if not fs:
        emit(tag=tag, llo="no final_bundles")
        return
    agg, bundles = collections.defaultdict(collections.Counter), collections.Counter()
    for ln in open(fs[-1], errors="replace").read().splitlines():
        m = BUNDLE_RE.match(ln)
        if not m:
            continue
        depth = len(m.group(2))
        bundles[depth] += 1
        for part in m.group(3).split(";;"):
            locs = LOC_RE.findall(part)
            mine = [x for x in locs if "sglang_tpu/" in x[0] or "dsa_k_prologue" in x[0]]
            loc = (mine or locs or [("noloc", "0")])[0]
            body = re.sub(r"/\*.*?\*/", "", part).strip()
            if "=" in body.split("[")[0]:
                body = body.split("=", 1)[1]
            t = TOK_RE.search(body)
            if not t:
                continue
            op = t.group(1)
            cat = ("mxu" if op.startswith("vmat") else "vld" if op.startswith("vld")
                   else "vst" if op.startswith("vst") else "v" if op.startswith("v") else "s")
            if cat == "vld":
                sm = re.search(r"sm:\$(0x[0-9a-f]+)", body)
                cat = f"vld_sm{sm.group(1) if sm else 'none'}"
            agg[(depth, os.path.basename(loc[0]), int(loc[1]))][cat] += 1
    k = max(bundles)  # innermost loop depth = the k loop
    emit(tag=tag, bundles_by_depth=dict(bundles),
         k_loop_ops=sum(sum(c.values()) for (d, _, _), c in agg.items() if d == k))
    top = sorted(((key, c) for key, c in agg.items() if key[0] == k),
                 key=lambda kv: -sum(kv[1].values()))[:14]
    for (d, f, ln), c in top:
        emit(tag=tag, depth=d, line=f"{f}:{ln}", n=sum(c.values()), cat=dict(c))


def mosaic_latent(new):
    for f in sorted(new):
        b = os.path.basename(f)
        if "infer-memref-layout" not in b:
            continue
        text = open(f, errors="replace").read()
        if "dsa_tensor_core" not in text and "vmem, tc" not in text:
            continue
        ts = sorted({t for t in re.findall(r"memref<2x16x[^<>]*xi8, #tpu\.tiled<[^>]*>", text)})
        if ts:
            emit(mosaic=b[-60:], vmem_u8_tiles=ts)


def main():
    import jax
    import jax.numpy as jnp

    from xplane_timeline import events
    from sglang_tpu.attention.pallas.source_dsa.kernels.dsa import fp8_attention as orig_mod
    from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.fp8_kv_cache import (
        SplitFP8KVCache, map_cache_parts,
    )
    from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.gather import _gather_split_fp8
    from sglang_tpu.attention.pallas.source_dsa.layers.attention import (
        dsa_sparse_backend_helpers as sh,
    )
    import fp8_attention_2d as new_mod

    f8 = jnp.float8_e4m3fn
    c = att.case("pr2198_extend_c1")
    P, T, K = c["P"], c["T"], c["topk"].shape[1]
    n = P * FP
    kv0, pe0 = jnp.asarray(c["prefix_kv"]), jnp.asarray(c["prefix_pe"])
    ql, qr, kvc, kpe = (jnp.asarray(c[k]) for k in ("q_latent", "q_rope", "kv_c", "k_pe"))
    kvl, pi, cuq = (jnp.asarray(c[k]) for k in ("kv", "pi", "cu_q"))
    S = c["S"]
    cukv = jnp.asarray((np.arange(S + 1) * c["ppr"] * FP).astype(np.int32))
    sm = float(c["sm_scale"])
    slots = jnp.asarray(to_physical(c["topk"], c["pi"], c["ppr"], c["cu_q"], page=FP))
    rows_bf = att.bf16_rows(c)
    bqs, bq, bk = 256, 16, 256
    nmb = T // bqs

    empty = SplitFP8KVCache(jnp.zeros((P, FP, 4, 128), jnp.uint8),
                            jnp.zeros((P, FP // 4, 4, 128), jnp.uint8))
    filled = jax.jit(lambda cc, kv_, pe_: sh._scatter_fused_kv_paged(
        cc, kv_, pe_, jnp.array([n], jnp.int32), jnp.arange(P, dtype=jnp.int32),
        jnp.array([0, n], jnp.int32), jnp.array([0, n], jnp.int32),
        kv_lora_rank=att.LAT, fp8_dtype=f8))(empty, kv0, pe0)
    written = jax.jit(lambda lat, rope: sh._scatter_fused_kv_paged(
        SplitFP8KVCache(lat, rope), kvc, kpe, kvl, pi, cuq, cukv,
        kv_lora_rank=att.LAT, fp8_dtype=f8))(filled.latent, filled.rope)
    cache = map_cache_parts(lambda x: x.reshape(-1, *x.shape[2:]), written)
    groups = jax.block_until_ready(jax.jit(lambda cc: tuple(
        _gather_split_fp8(cc, slots[i * bqs:(i + 1) * bqs].reshape(-1))
        for i in range(nmb)))(cache))
    counts = jnp.full((T,), K, jnp.int32)

    def make(mod):
        def pre(*gs):
            outs = []
            for i, g in enumerate(gs):
                cur = map_cache_parts(lambda x: x.reshape(1, bqs, K, *x.shape[1:]), g)
                outs.append(mod.fp8_tensor_core_attention(
                    ql[i * bqs:(i + 1) * bqs], qr[i * bqs:(i + 1) * bqs], cur,
                    counts[i * bqs:(i + 1) * bqs], jnp.int32(0), sm_scale=sm, bq=bq,
                    b_topk=bk, fp8_dtype=f8, fp8_pv=True))
            return jnp.concatenate(outs, axis=0)
        return pre

    variants = {"orig": (orig_mod, "dsa_tensor_core_attention_fp8_qkv.1"),
                "tile2d": (new_mod, "dsa_tensor_core_attention_fp8_qkv_2d")}
    outs, jits = {}, {}
    for tag, (mod, kname) in variants.items():
        try:
            fn = make(mod)
            fn.__name__ = f"t2d_{tag}"
            before, mbefore = files(DUMP), files(MOSAIC)
            jf = jax.jit(fn)
            out = jax.block_until_ready(jf(*groups))
            outs[tag], jits[tag] = np.asarray(out), jf
            r = dict(variant=tag, accuracy_vs_bf16=att.error_vs_bf16(
                c, outs[tag].astype(np.float64), rows_bf))
            if tag != "orig" and "orig" in outs:
                r["exact_vs_orig"] = bool(np.array_equal(outs[tag].view(np.uint16),
                                                         outs["orig"].view(np.uint16)))
                r["max_abs_diff_vs_orig"] = float(np.max(np.abs(
                    outs[tag].astype(np.float32) - outs["orig"].astype(np.float32))))
            emit(**r)
            llo_lines(tag, files(DUMP) - before, kname)
            mosaic_latent(files(MOSAIC) - mbefore)
        except Exception as e:  # report, keep the other variant
            import traceback

            traceback.print_exc()
            emit(variant=tag, fail=repr(e)[:1500])

    # Device module time: 3 captures per variant, one call each.
    for tag, jf in jits.items():
        us = []
        for k in range(3):
            d = f"/tmp/t2d_trace/{tag}_{k}"
            jax.block_until_ready(jf(*groups))
            jax.profiler.start_trace(d)
            try:
                jax.block_until_ready(jf(*groups))
            finally:
                jax.profiler.stop_trace()
            ev = events(d)
            mods = [(s, e) for p, ln, nm, s, e in ev
                    if " SparseCore" not in p and ln == "XLA Modules" and f"t2d_{tag}" in nm]
            attn = sorted((s, e) for p, ln, nm, s, e in ev
                          if " SparseCore" not in p and ln == "XLA Ops"
                          and "dsa_tensor_core_attention" in nm)
            if mods:
                us.append(round((mods[0][1] - mods[0][0]) / 1e6, 1))
            if k == 0:
                emit(variant=tag, attn_chunks_us=[round((e - s) / 1e6, 1) for s, e in attn])
        emit(variant=tag, module_us=us,
             us_per_query=round(sorted(us)[len(us) // 2] / T, 3) if us else None)
    print("T2D_DONE", flush=True)


if __name__ == "__main__":
    main()
