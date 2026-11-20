"""Test only: Source / Target TensorCore attention LLO, per source line.

Compiles (does not time) the same two kernels as bench_attn_llo.py with LLO
source annotations on, then for each kernel's final_bundles file prints:

* ``SRCLINE``: per loop depth and per Python source line, instruction counts by
  category (vld, vst, mxu, reorder, vmul, softmax, other vector, scalar), plus
  the source text of that line. Each instruction is attributed to the first
  ``sglang_tpu`` frame of its ``loc(...)`` annotation.
* ``ALLOC``: the VMEM allocation declarations of the kernel (shape, tiling,
  size), to check the scratch tile layout.
* ``MOSAIC``: with ``LLO_MOSAIC=1``, the distinct memref types in the Mosaic
  dump files (tiled layouts after the layout passes).

Kernel and production code are untouched.
"""

import collections
import glob
import json
import os
import re
import sys

DUMP = "/tmp/llo_dump"
MOSAIC = "/tmp/mosaic_dump"
os.makedirs(DUMP, exist_ok=True)
os.makedirs(MOSAIC, exist_ok=True)
os.environ.setdefault("JAX_PLATFORMS", "tpu")
_flags = (f" --xla_jf_dump_to={DUMP} --xla_jf_dump_llo_text=true"
          " --xla_mosaic_enable_llo_source_annotations=true")
if os.environ.get("LLO_MOSAIC") == "1":
    _flags += f" --xla_mosaic_dump_to={MOSAIC}"
# --xla_jf_* must go to LIBTPU_INIT_ARGS: they abort the XLA_FLAGS parser.
os.environ["LIBTPU_INIT_ARGS"] = (os.environ.get("LIBTPU_INIT_ARGS", "") + _flags).strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

FP, P, K, H = 64, 801, 2048, 64
BUNDLE_RE = re.compile(r"\s*(0x[0-9a-f]+)\s*:\s*(>*)\s*\{(.*)\}\s*$")
TOK_RE = re.compile(r"(?<![%\w.$#])([a-z][a-z0-9_]*(?:\.[\w]+)*)")
LOC_RE = re.compile(r'"([^"]+\.py)":(\d+)')
REORDER = ("vshrl", "vshll", "vpack", "vunpack", "vcombine", "vrot", "vperm", "vmpackc",
           "vpop.permute", "vand", "vor", "vxor", "vsel", "vmov", "vbcast", "vnez")
SOFTMAX = ("vmax", "vsub", "vexp", "vadd", "vpop.eup", "vrcp", "vpop.xlane", "vcmp",
           "vclamp", "vmaxabs", "vmovacc")


def files_now():
    return {f for f in glob.glob(f"{DUMP}/**/*", recursive=True) if os.path.isfile(f)}


def category(op):
    if op.startswith("vmat") or op.startswith("vpop.f32.mrb"):
        return "mxu"
    if op.startswith("vld"):
        return "vld"
    if op.startswith("vst"):
        return "vst"
    if op.startswith("vmul"):
        return "vmul"
    if op.startswith(REORDER):
        return "reorder"
    if op.startswith(SOFTMAX):
        return "softmax"
    if op.startswith("v"):
        return "vother"
    return "scalar"


def parse_part(part):
    locs = LOC_RE.findall(part)
    mine = [x for x in locs if "sglang_tpu/" in x[0] or "dsa_k_prologue" in x[0]]
    loc = (mine or locs or [("noloc", "0")])[0]
    body = re.sub(r"/\*.*?\*/", "", part).strip()
    if "=" in body.split("[")[0]:
        body = body.split("=", 1)[1]
    m = TOK_RE.search(body)
    return (m.group(1) if m else None), (os.path.basename(loc[0]), int(loc[1]), loc[0])


_src_cache = {}


def src_text(path, line):
    if path not in _src_cache:
        try:
            _src_cache[path] = open(path, errors="replace").read().splitlines()
        except OSError:
            _src_cache[path] = []
    lines = _src_cache[path]
    return lines[line - 1].strip()[:120] if 0 < line <= len(lines) else ""


# The tile-read source line of each kernel: Source fp8_attention.py:194, Target kernel.py:396.
VLD_LINES = {"src": ("fp8_attention.py", 194), "tgt": ("kernel.py", 396)}
SM_RE = re.compile(r"\bsm:\$?(0x[0-9a-f]+|\d+)")
SS_RE = re.compile(r"\bss:\$?(0x[0-9a-f]+|-?\d+)")
ADDR_RE = re.compile(r"\[vmem:\[([^\]]*)\]")


def vld_detail(tag, lines):
    """Every vld attributed to the tile-read line: sublane-mask / stride histogram,
    how many distinct VMEM addresses they read (re-reads show as repeats), and
    a raw sample, so the per-row vld count can be accounted for exactly."""
    want = VLD_LINES.get(tag)
    if want is None:
        return
    raw, masks, strides, addrs, depths = [], collections.Counter(), collections.Counter(), \
        collections.Counter(), collections.Counter()
    for ln in lines:
        m = BUNDLE_RE.match(ln)
        if not m:
            continue
        for part in m.group(3).split(";;"):
            op, (fname, lno, _) = parse_part(part)
            if not op or not op.startswith("vld") or (fname, lno) != want:
                continue
            body = re.sub(r"/\*.*?\*/", "", part).strip()
            raw.append(body)
            depths[len(m.group(2))] += 1
            sm, ss, ad = SM_RE.search(body), SS_RE.search(body), ADDR_RE.search(body)
            masks[sm.group(1) if sm else "none"] += 1
            strides[ss.group(1) if ss else "none"] += 1
            addrs[ad.group(1) if ad else "?"] += 1
    print("VLDSTAT " + json.dumps({
        "tag": tag, "line": f"{want[0]}:{want[1]}", "n": len(raw), "by_depth": dict(depths),
        "sublane_masks": masks.most_common(12), "strides": strides.most_common(12),
        "distinct_addrs": len(addrs), "max_repeat": max(addrs.values()) if addrs else 0,
        "addr_repeat_hist": collections.Counter(addrs.values()).most_common(8),
        "opcodes": collections.Counter(TOK_RE.search(r.split("=", 1)[-1]).group(1)
                                       for r in raw).most_common(6)}), flush=True)
    for r in raw[:48]:
        print(f"VLDRAW {tag}|{r[:260]}", flush=True)


def report(tag, new_files, key):
    fs = [f for f in sorted(new_files) if key in os.path.basename(f)
          and os.path.basename(f).endswith("final_bundles.txt")
          and "schedule-analysis" not in os.path.basename(f)]
    if not fs:
        print("SRCLINE " + json.dumps({"tag": tag, "error": "no final_bundles"}), flush=True)
        return
    f = fs[-1]
    text = open(f, errors="replace").read()
    lines = text.splitlines()
    # (depth, file, line) -> Counter(category); also bundles per depth.
    agg = collections.defaultdict(collections.Counter)
    paths = {}
    bundles = collections.Counter()
    noloc = 0
    for ln in lines:
        m = BUNDLE_RE.match(ln)
        if not m:
            continue
        depth = len(m.group(2))
        bundles[depth] += 1
        for part in m.group(3).split(";;"):
            op, (fname, lno, full) = parse_part(part)
            if op is None:
                continue
            noloc += fname == "noloc"
            agg[(depth, fname, lno)][category(op)] += 1
            paths[(fname, lno)] = full
    vld_detail(tag, lines)
    print("SRCHEAD " + json.dumps({"tag": tag, "file": os.path.basename(f),
                                    "bundles_by_depth": dict(bundles), "noloc_ops": noloc}),
          flush=True)
    for (depth, fname, lno), c in sorted(agg.items(), key=lambda kv: (kv[0][0], -sum(kv[1].values()))):
        if sum(c.values()) < 8:
            continue
        print("SRCLINE " + json.dumps({
            "tag": tag, "depth": depth, "file": fname, "line": lno, "n": sum(c.values()),
            "cat": dict(c), "src": src_text(paths[(fname, lno)], lno)}), flush=True)
    allocs = [x.strip() for x in lines if "#allocation" in x and "vmem" in x.lower()
              and ("shape" in x or "size" in x) and not BUNDLE_RE.match(x)]
    for a in allocs[:60]:
        print("ALLOC " + json.dumps({"tag": tag, "decl": re.sub(r"/\*.*?\*/", "", a)[:300]}),
              flush=True)


def mosaic_layouts():
    fs = sorted(f for f in glob.glob(f"{MOSAIC}/**/*", recursive=True) if os.path.isfile(f))
    print("MOSAICFILES " + json.dumps([os.path.basename(f) for f in fs][:80]), flush=True)
    # File names are "anonymous"; tell kernels apart by content. Only the two
    # layout passes: memref tiling of every ref, and the layout of vector loads.
    for f in fs:
        base = os.path.basename(f)
        if not any(p in base for p in ("infer-memref-layout", "infer-vector-layout")):
            continue
        text = open(f, errors="replace").read()
        tag = ("src" if "dsa_tensor_core_attention" in text
               else "tgt" if "MLA-" in text else "unk")
        if "infer-memref-layout" in base:
            types = []
            for t in re.findall(r"memref<[^<>]*(?:<[^<>]*(?:<[^<>]*>[^<>]*)*>[^<>]*)*>", text):
                if t not in types:
                    types.append(t)
            print("MOSAIC " + json.dumps({"tag": tag, "pass": "memref", "file": base[-70:],
                                          "memrefs": [t[:260] for t in types[:60]]}), flush=True)
        else:
            loads = [ln.strip()[:500] for ln in text.splitlines()
                     if ("vector.load" in ln or "tpu.load" in ln or "vector_load" in ln)
                     and "layout" in ln]
            print("MOSAIC " + json.dumps({"tag": tag, "pass": "vector", "file": base[-70:],
                                          "n_loads": len(loads), "loads": loads[:25]}), flush=True)


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sglang_tpu.attention.pallas.mla_sparse import dsa_gather, kv_cache_utils as kcu
    from sglang_tpu.attention.pallas.mla_sparse.kernel import (
        prepare_q_inputs, sparse_attention_chunk,
    )
    from sglang_tpu.attention.pallas.source_dsa.kernels.dsa import exact_attention as ea
    from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.fp8_kv_cache import (
        SplitFP8KVCache, map_cache_parts,
    )
    from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.gather import _gather_split_fp8

    rng = np.random.default_rng(0)
    f8 = jnp.float8_e4m3fn
    sm = 0.04

    # Source: one microbatch, tiles (256, 16, 256), fp8-qk-pv.
    bqs, bq, bk = 256, 16, 256
    cache = SplitFP8KVCache(jnp.zeros((P * FP, 4, 128), jnp.uint8),
                            jnp.zeros((P * FP // 4, 4, 128), jnp.uint8))
    slots = jnp.asarray(rng.integers(0, P * FP, (bqs * K,), dtype=np.int32))
    g = jax.block_until_ready(jax.jit(_gather_split_fp8)(cache, slots))
    ql = jnp.asarray(rng.standard_normal((bqs, H, 512)), jnp.bfloat16)
    qr = jnp.asarray(rng.standard_normal((bqs, H, 64)), jnp.bfloat16)
    counts = jnp.full((bqs,), K, jnp.int32)

    def llo_src(ql_, qr_, g_, counts_):
        cur = map_cache_parts(lambda x: x.reshape(1, bqs, K, *x.shape[1:]), g_)
        return ea._tensor_core_attention(ql_, qr_, cur, counts_, jnp.int32(0), sm_scale=sm,
                                         bq=bq, b_topk=bk, fp8_dtype=f8, fp8_q=True,
                                         fp8_pv=True)

    before = files_now()
    jax.jit(llo_src).lower(ql, qr, g, counts).compile()
    report("src", files_now() - before, "dsa_tensor_core_attention_fp8_qkv")

    # Target: one 64-query chunk, per-query kernel, W8A16.
    T = 64
    no = jnp.zeros((P, FP, 4, 128), jnp.uint8)
    ro = jnp.zeros((P, FP // 4, 4, 128), jnp.uint8)
    phys = jnp.asarray(rng.integers(0, P * FP, (T, K), dtype=np.int32))
    gn, gr = jax.block_until_ready(jax.jit(lambda a, b, i: dsa_gather.dsa_gather(
        a, b, i.reshape(-1), num_valid_indices=jnp.int32(T * K)))(no, ro, phys))
    tile = (gn.reshape(T, K * kcu.WORD_BYTES, kcu.TILE_LANE_BYTES), gr.reshape(T, K, -1))
    q = prepare_q_inputs(jnp.concatenate([ql[:T], qr[:T]], axis=-1))
    kv_lens = jnp.full((T,), K, jnp.int32)

    def llo_tgt(q_, no_, ro_, kvl_, phys_, gn_, gr_):
        return sparse_attention_chunk(q_, no_, ro_, kvl_, phys_, None, start_pos=0, end_pos=T,
                                      token_end=jnp.int32(T), sm_scale=sm, k_scale=1.0,
                                      topk_override=K, gathered_group=(gn_, gr_))

    before = files_now()
    jax.jit(llo_tgt).lower(q, no, ro, kv_lens, phys, *tile).compile()
    report("tgt", files_now() - before, "MLA-")
    if os.environ.get("LLO_MOSAIC") == "1":
        mosaic_layouts()
    print("LLO_DONE", flush=True)


if __name__ == "__main__":
    main()
