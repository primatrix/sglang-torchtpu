"""Test only: LLO bundles of the Source and Target TensorCore attention kernels.

Compiles (does not time) one Source microbatch (fp8-qk-pv, tiles 256/16/256)
and one Target chunk (64 queries, per-query kernel, W8A16) on pre-gathered
tiles, with the LLO text dump on. For each kernel's bundle files it prints:

* ``LLOSTAT``: bundle counts, how many bundles carry an MXU op (``vmat*``), a
  vector ALU op (any other ``v*`` op except loads / stores), both, or nothing,
  plus an opcode histogram so the classification can be checked.
* ``LLOOP``: the final bundle listing reduced to address + opcodes, so loop
  bodies can be read from ``falcon exp logs`` (the artifact bucket is not
  readable from outside).
"""

import collections
import glob
import json
import os
import re
import sys

DUMP = "/tmp/llo_dump"
os.makedirs(DUMP, exist_ok=True)
os.environ.setdefault("JAX_PLATFORMS", "tpu")
# --xla_jf_* must go to LIBTPU_INIT_ARGS: they abort the XLA_FLAGS parser.
os.environ["LIBTPU_INIT_ARGS"] = (
    os.environ.get("LIBTPU_INIT_ARGS", "")
    + f" --xla_jf_dump_to={DUMP} --xla_jf_dump_llo_text=true"
).strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

FP, P, K, H = 64, 801, 2048, 64
OP_RE = re.compile(r"(?:=\s*|^\s*)([a-z][\w.]*)")


def files_now():
    return {f for f in glob.glob(f"{DUMP}/**/*", recursive=True) if os.path.isfile(f)}


def opcodes(bundle):
    ops = []
    for part in bundle.split(";;"):
        part = re.sub(r"/\*.*?\*/", "", part).strip()
        m = OP_RE.search(part)
        if m:
            ops.append(m.group(1))
    return ops


def kind(op):
    if op.startswith("vmat"):
        return "mxu"
    if op.startswith(("vld", "vst")):
        return "mem"
    if op.startswith("v"):
        return "vpu"
    return "other"


def report(tag, new_files, key):
    names = sorted(new_files)
    print("LLOFILES " + json.dumps({"tag": tag, "files": [os.path.basename(f) for f in names]}),
          flush=True)
    bundle_files = [f for f in names if "bundles" in os.path.basename(f)
                    and key in open(f, errors="replace").read(400000)]
    final = None
    for f in bundle_files:
        lines = open(f, errors="replace").read().splitlines()
        rows = []
        for ln in lines:
            m = re.match(r"\s*(0x[0-9a-f]+)\s*:\s*>?\s*\{(.*)\}\s*$", ln)
            if m:
                rows.append((m.group(1), opcodes(m.group(2))))
        if not rows:
            continue
        hist = collections.Counter(op for _, ops in rows for op in ops)
        kinds = [set(kind(o) for o in ops) for _, ops in rows]
        stat = {
            "tag": tag, "file": os.path.basename(f), "bundles": len(rows),
            "empty": sum(not k for k in kinds),
            "mxu": sum("mxu" in k for k in kinds),
            "vpu": sum("vpu" in k for k in kinds),
            "mxu_and_vpu": sum({"mxu", "vpu"} <= k for k in kinds),
            "mxu_only": sum("mxu" in k and "vpu" not in k for k in kinds),
            "vpu_only": sum("vpu" in k and "mxu" not in k for k in kinds),
            "mem_only": sum(k == {"mem"} for k in kinds),
            "top_ops": hist.most_common(60),
        }
        print("LLOSTAT " + json.dumps(stat), flush=True)
        final = (f, rows)  # files sort by pass number; keep the last
    if final is None:
        print("LLOSTAT " + json.dumps({"tag": tag, "error": "no bundle file matched"}), flush=True)
        return
    finals[tag] = final


def listing(tag, cap):
    # `falcon exp logs` returns at most 5000 lines: per-64-bundle window counts
    # for the whole file, then at most `cap` bundles of the densest MXU stretch.
    f, rows = finals[tag]
    kinds = [set(kind(o) for o in ops) for _, ops in rows]
    for s in range(0, len(rows), 64):
        w = kinds[s:s + 64]
        print("LLOWIN " + json.dumps({
            "tag": tag, "start": rows[s][0], "n": len(w),
            "mxu": sum("mxu" in k for k in w), "vpu": sum("vpu" in k for k in w),
            "both": sum({"mxu", "vpu"} <= k for k in w), "empty": sum(not k for k in w),
            "other": sorted({o for _, ops in rows[s:s + 64] for o in ops
                             if kind(o) == "other"})[:12]}), flush=True)
    mx = [1 if "mxu" in k else 0 for k in kinds]
    best, lo = -1, 0
    for s in range(0, max(1, len(rows) - cap + 1), 64):
        c = sum(mx[s:s + cap])
        if c > best:
            best, lo = c, s
    print(f"LLOOP_BEGIN {tag} {os.path.basename(f)} {len(rows)} from {lo}", flush=True)
    for addr, ops in rows[lo:lo + cap]:
        print(f"LLOOP {tag}|{addr} {' '.join(ops)}", flush=True)
    print(f"LLOOP_END {tag}", flush=True)


finals = {}


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

    # ---------------- Source: one microbatch, (256, 16, 256) ----------------
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
    report("src", files_now() - before, "dsa_tensor_core_attention")

    # ---------------- Target: one 64-query chunk, per-query kernel ----------------
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
    for tag, cap in (("src", 2600), ("tgt", 900)):
        if tag in finals:
            listing(tag, cap)
    print("LLO_DONE", flush=True)


if __name__ == "__main__":
    main()
