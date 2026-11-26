"""Test only: MXU / VPU opcodes of the vllm-torchtpu sparse MLA attention kernel.

Compiles (does not time) the vllm kernel from ``vllm_mla/`` on the PR 2198
shapes with the LLO text dump on, for the SparseCore (vllm default) and
TensorCore cache layouts. For every final-bundle file of the attention kernel
(``MLA-`` in the name) it prints one ``LLOSTAT`` line: bundle count and the
count of each MXU / conversion / relayout opcode, so we can read

* whether QK / PV push FP8 (``vmatpush*.f8e4m3``) or BF16 (``vmatpush*.bf16``),
* how many ``vmatmul`` the FP32-LHS PV issues per block (passes),
* how many ``vcvt`` / ``vunpack`` / relayout ops the KV unpack costs.

The kernel body handles one attention batch (16 queries by default), so static
counts are per batch; ``batch_size`` is printed for the per-query division.
"""

import argparse
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import bench_fp8_attention_ab as att  # noqa: E402

FP = att.FP = 64
BUNDLE_RE = re.compile(r"\s*(0x[0-9a-f]+)\s*:\s*(>*)\s*\{(.*)\}\s*$")
TOK_RE = re.compile(r"(?<![%\w.$#])([a-z][a-z0-9_]*(?:\.[\w]+)*)")
KEEP = ("vmat", "vpop", "vcvt", "vunpack", "vpack", "vperm", "vrot", "vcombine",
        "vshl", "vshr", "vand", "vor", "vsel", "vcmp", "vexp", "vpow", "vrcp", "vmax", "vadd",
        "vmul", "vsub", "vld", "vst")


def opcodes(bundle):
    ops = []
    for part in bundle.split(";;"):
        part = re.sub(r"/\*.*?\*/", "", part).strip()
        if "=" in part.split("[")[0]:
            part = part.split("=", 1)[1]
        m = TOK_RE.search(part)
        if m:
            ops.append(m.group(1))
    return ops


def files_now():
    return {f for f in glob.glob(f"{DUMP}/**/*", recursive=True) if os.path.isfile(f)}


def report(tag, new_files):
    names = sorted(f for f in new_files if f.endswith("final_bundles.txt"))
    print("LLOFILES " + json.dumps({"tag": tag, "files": [os.path.basename(f)[-90:] for f in names]}),
          flush=True)
    for f in names:
        base = os.path.basename(f)
        if "MLA-" not in base:
            continue
        rows = []
        for ln in open(f, errors="replace"):
            m = BUNDLE_RE.match(ln)
            if m:
                rows.append((opcodes(m.group(3)), len(m.group(2))))
        hist = collections.Counter(o for ops, _ in rows for o in ops)
        keep = {o: n for o, n in sorted(hist.items()) if o.startswith(KEEP)}
        depth = collections.Counter(d for _, d in rows)
        mxu_b = sum(any(o.startswith("vmat") for o in ops) for ops, _ in rows)
        print("LLOSTAT " + json.dumps({
            "tag": tag, "file": base[-90:], "bundles": len(rows), "mxu_bundles": mxu_b,
            "bundles_by_depth": dict(sorted(depth.items())), "ops": keep}), flush=True)
        # A few raw MXU lines so the opcode spelling can be checked by eye.
        raw = [re.sub(r"/\*.*?\*/", "", ln).strip()[:400]
               for ln in open(f, errors="replace") if "vmat" in ln][:12]
        print("LLORAW " + json.dumps({"tag": tag, "file": base[-60:], "lines": raw}), flush=True)


def report_order(tag, new_files):
    """Walk the TC attention MLA bundle file in program order and show how the
    MXU (vmat/vpop) work interleaves with reorder (vunpack/vperm...) vs softmax
    (vexp/vmax/vrcp) VPU work: per-bundle co-issue counts + a run-length tag
    sequence. Answers whether PV overlaps the reorder or the softmax."""
    names = sorted(
        f for f in new_files
        if f.endswith("final_bundles.txt") and "MLA-" in os.path.basename(f)
    )
    for f in names:
        base = os.path.basename(f)
        seq = []
        n_mxu = co_reorder = co_softmax = mxu_alone = 0
        for ln in open(f, errors="replace"):
            m = BUNDLE_RE.match(ln)
            if not m:
                continue
            ops = opcodes(m.group(3))
            mxu = any(o.startswith(("vmat", "vpop")) for o in ops)
            reorder = any(
                o.startswith(("vunpack", "vperm", "vrot", "vcombine", "vshrl", "vshr"))
                for o in ops
            )
            softmax = any(o.startswith(("vexp", "vmax", "vrcp")) for o in ops)
            if mxu:
                n_mxu += 1
                co_reorder += reorder
                co_softmax += softmax
                mxu_alone += not (reorder or softmax)
                seq.append("M")
            elif reorder:
                seq.append("R")
            elif softmax:
                seq.append("S")
            else:
                seq.append(".")
        rle = []
        for ch in seq:
            if rle and rle[-1][0] == ch:
                rle[-1][1] += 1
            else:
                rle.append([ch, 1])
        rle_str = " ".join(f"{c}{n}" for c, n in rle)
        print("LLOORDER " + json.dumps({
            "tag": tag, "file": base[-70:], "bundles": len(seq), "mxu": n_mxu,
            "mxu_coissue_reorder": co_reorder, "mxu_coissue_softmax": co_softmax,
            "mxu_alone": mxu_alone, "rle": rle_str[:12000]}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=["pr2198_extend_c1", "pr2198_verify_b4"])
    p.add_argument("--layouts", nargs="+", default=["sparsecore", "tensorcore"])
    p.add_argument("--order", action="store_true", help="also dump ordered MXU/reorder/softmax interleave")
    a = p.parse_args()

    import jax
    import jax.numpy as jnp

    from vllm_mla.mla import kv_cache_utils as kcu
    from vllm_mla.mla.sparse import kernel as kmod

    print("LLOINFO " + json.dumps({"jax": jax.__version__, "kernel": kmod.__file__,
                                   "batch_size_default": getattr(kmod, "DEFAULT_BATCH_SIZE", None)}),
          flush=True)
    f8 = jnp.float8_e4m3fn
    fmax = float(jnp.finfo(f8).max)
    q8 = lambda x: jnp.clip(x.astype(jnp.float32), -fmax, fmax).astype(f8)

    for name in a.cases:
        c = att.case(name)
        P = c["P"]
        q = jnp.concatenate([jnp.asarray(c["q_latent"]), jnp.asarray(c["q_rope"])], axis=-1)
        kvc, kpe = q8(jnp.asarray(c["kv_c"])), q8(jnp.asarray(c["k_pe"]))
        kvl, pi, cuq = (jnp.asarray(c[k]) for k in ("kv", "pi", "cu_q"))
        S = c["S"]
        dist = jnp.asarray(np.array([S] * 3 if c["mode"] == "decode" else [0, 0, S], np.int32))
        topk = jnp.asarray(c["topk"])
        sm = float(c["sm_scale"])
        for layout in a.layouts:
            tag = f"{name}_{layout}"
            try:
                lay = {"tensorcore": kcu.KVCacheLayout.TENSORCORE,
                       "sparsecore": kcu.KVCacheLayout.SPARSECORE}[layout]
                ns = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.NOPE, lay, P, FP, 512)
                rs = kcu.SparseMLAKVCacheSpec.create(kcu.KVCacheType.ROPE, lay, P, FP, 64)

                def step(nope, rope, q_, kvc_, kpe_, _ns=ns, _rs=rs, _l=layout):
                    active = jnp.minimum(cuq, cuq[dist[2]])
                    nope, rope = kcu.update_sparse_mla_kv_cache(
                        nope, rope, kvc_, kpe_, kvl, pi, active, nope_spec=_ns, rope_spec=_rs)
                    out = kmod.sparse_ragged_paged_attention(
                        q_, nope, rope, topk, pi, cuq, dist,
                        sm_scale=sm, k_scale=1.0, cache_layout=_l)
                    return out[..., :512], nope, rope

                args = (jax.ShapeDtypeStruct(ns.shape, ns.jax_dtype),
                        jax.ShapeDtypeStruct(rs.shape, rs.jax_dtype), q, kvc, kpe)
                before = files_now()
                jax.jit(step).lower(*args).compile()
                new = files_now() - before
                report(tag, new)
                if a.order:
                    report_order(tag, new)
            except Exception as e:  # report, keep the other variants
                import traceback

                traceback.print_exc()
                print("LLOFAIL " + json.dumps({"tag": tag, "error": repr(e)[:800]}), flush=True)
    print("LLO_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
