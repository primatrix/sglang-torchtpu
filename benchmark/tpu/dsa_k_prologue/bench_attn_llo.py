"""Test only: what the SparseCore-layout attention kernel spends its bundles on.

The pre-gathered profile (exp-q9c5apcd9d) put the 16 ``MLA-sparsecore`` kernels at
2.56 ms of a 2.81 ms extend_c1 call, at ~228 TFLOP/s and ~1.05 TB/s: neither the
MXU nor HBM is near its limit. This compiles one chunk's kernel (#2246 config:
chunk 128, batch 16, 64 heads, head_dim 576 -> 640, top-k 2048, pre-gathered
buffers so no SparseCore gather is in the program) with the LLO text dump on,
and summarises the final bundles on the pod (the dump is too large for the log):

* ``LLO_SHAPE``: bundles, empty bundles, ops per non-empty bundle;
* ``LLO_OPS``: op count by mnemonic (``vmatmul``/``vmatpush`` = MXU, ``vld``/``vst``
  = VMEM load/store, ``vexp``/``vmax``/``vadd``/``vmul``/``vsel``/``vpack`` ... =
  VPU, ``dma`` ...);
* ``LLO_SRC``: op count by ``kernel.py`` source line, with its mnemonic mix, so
  each cost maps to a line of the kernel;
* ``LLO_LABEL``: non-bundle lines (labels, loop markers), to see the loop nest;
* ``LLO_HOT``: the longest straight run of bundles that have an MXU op, printed
  compactly, as a sample of the steady-state body.

Static counts are not weighted by trip count; ``LLO_SRC`` per line together with
the loop markers shows which lines sit in the inner loop.
"""

import collections
import glob
import json
import os
import re
import sys

DUMP = "/tmp/attn_llo"
os.makedirs(DUMP, exist_ok=True)
os.environ.setdefault("JAX_PLATFORMS", "tpu")
os.environ["LIBTPU_INIT_ARGS"] = (
    os.environ.get("LIBTPU_INIT_ARGS", "")
    + f" --xla_jf_dump_to={DUMP} --xla_jf_dump_llo_text=true"
    " --xla_mosaic_enable_llo_source_annotations=true"
).strip()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LOC = re.compile(r'/\*\s*loc\("([^"]+)":(\d+)')
KLINE = re.compile(r'loc\("[^"]*mla_sparse/kernel\.py":(\d+)')


def mnemonic(op):
    op = op.split("/*")[0].strip()
    if not op:
        return None
    body = op.split("=", 1)[1].strip() if "=" in op.split("[")[0] else op
    tok = body.split()[0] if body.split() else ""
    tok = tok.lstrip(">").strip()
    return tok or None


def analyse(path):
    lines = open(path, errors="replace").read().splitlines()
    bundles, labels = [], []
    for ln in lines:
        m = re.match(r"\s*(0x[0-9a-f]+)\s*:\s*>?\s*(\{.*\})\s*$", ln)
        if m:
            bundles.append(m.group(2)[1:-1])
        elif ln.strip():
            labels.append(ln.strip()[:160])
    ops_total = collections.Counter()
    base = collections.Counter()
    src = collections.defaultdict(collections.Counter)
    per_bundle, mxu_flags = [], []
    for b in bundles:
        ops = [o for o in b.split(";;") if o.strip()]
        per_bundle.append(len(ops))
        has_mxu = False
        for o in ops:
            mn = mnemonic(o)
            if not mn:
                continue
            ops_total[mn] += 1
            bn = mn.split(".")[0]
            base[bn] += 1
            if "matmul" in bn or "matpush" in bn or "matres" in bn or bn.startswith("vmat"):
                has_mxu = True
            k = KLINE.search(o)
            src[int(k.group(1)) if k else -1][bn] += 1
        mxu_flags.append(has_mxu)
    nonempty = [n for n in per_bundle if n]
    print("LLO_SHAPE " + json.dumps({
        "file": os.path.basename(path)[-80:], "bundles": len(bundles),
        "empty": sum(1 for n in per_bundle if n == 0),
        "ops": sum(per_bundle),
        "ops_per_nonempty": round(sum(nonempty) / max(1, len(nonempty)), 2),
        "mxu_bundles": sum(mxu_flags),
        "hist_ops_per_bundle": dict(sorted(collections.Counter(per_bundle).items())),
    }), flush=True)
    print("LLO_OPS " + json.dumps(base.most_common(40)), flush=True)
    print("LLO_OPS_FULL " + json.dumps(ops_total.most_common(60)), flush=True)
    rows = sorted(src.items(), key=lambda kv: -sum(kv[1].values()))
    for line, c in rows[:40]:
        print("LLO_SRC " + json.dumps({"line": line, "ops": sum(c.values()),
                                       "mix": c.most_common(8)}), flush=True)
    print("LLO_LABELS " + str(len(labels)), flush=True)
    for x in labels[:120]:
        print("LLO_LABEL " + x, flush=True)
    # Longest run of consecutive bundles where at least every other has an MXU op.
    best, cur_s, cur_n = (0, 0), None, 0
    for i, f in enumerate(mxu_flags):
        if f:
            if cur_s is None:
                cur_s, cur_n = i, 0
            cur_n = i - cur_s + 1
            if cur_n > best[1]:
                best = (cur_s, cur_n)
        elif cur_s is not None and i - cur_s - cur_n > 2:
            cur_s = None
    s, n = best
    print("LLO_HOT " + json.dumps({"start": s, "len": n}), flush=True)
    for b in bundles[s:s + min(n, 120)]:
        ops = [mnemonic(o) or "" for o in b.split(";;") if o.strip()]
        ks = sorted({int(k) for k in KLINE.findall(b)})
        print("LLO_B " + " | ".join(ops)[:220] + f"   @{ks[:4]}", flush=True)


def main():
    import jax
    import jax.numpy as jnp

    from sglang_tpu.attention.pallas.mla_sparse import kv_cache_utils as kcu
    from sglang_tpu.attention.pallas.mla_sparse import native_sc_gather as nsc
    from sglang_tpu.attention.pallas.mla_sparse.kernel import sparse_ragged_paged_attention

    T, H, TOPK, P, FP = 128, 64, 2048, 801, 64
    lay = kcu.KVCacheLayout.SPARSECORE
    ns = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.NOPE, lay, P, FP, 512)
    rs = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.ROPE, lay, P, FP, 64)
    S = jax.ShapeDtypeStruct
    args = (S((T, H, 576), jnp.bfloat16), S(ns.shape, ns.jax_dtype), S(rs.shape, rs.jax_dtype),
            S((T, TOPK), jnp.int32), (S((T, TOPK, 128), jnp.uint32),),
            (S((T, TOPK // nsc.ROPE_PACKING, 128), jnp.uint32),))
    f = jax.jit(lambda q, n, r, rows, gn, gr: sparse_ragged_paged_attention(
        q, n, r, rows, None, jnp.array([0, T], jnp.int32), jnp.array([0, 0, 1], jnp.int32),
        sm_scale=0.1, k_scale=1.0, cache_layout="sparsecore",
        gather_and_attention_chunk_size=128, attention_kernel_batch_size=16,
        sparsecore_atoms_per_batch=16, pregathered=(gn, gr)))
    f.lower(*args).compile()
    files = sorted(x for x in glob.glob(f"{DUMP}/**/*", recursive=True) if os.path.isfile(x))
    print(f"LLO_FILES {len(files)}", flush=True)
    mine = [x for x in files if "MLA-sparsecore" in os.path.basename(x)]
    for x in mine[:40]:
        print(f"LLO_FILE {os.path.getsize(x)} {os.path.basename(x)[-110:]}", flush=True)
    final = [x for x in mine if "packed-bundles-no-spills-fills" in x] or \
            [x for x in mine if "final_bundles" in x]
    for x in final[-1:]:
        analyse(x)
    print("LLO_DONE", flush=True)


if __name__ == "__main__":
    main()
