"""Test only: DSA NoPE KV 4x4 byte transpose, compiler reshape vs vperm.slane.

The vllm-torchtpu sparse MLA kernel (SparseCore cache layout) turns each
64-token chunk of gathered NoPE KV from ``u32[64, 128]`` (token t = one row,
word w byte g = feature g*128+w) into ``fp8[64, 512]`` (one token per row, MXU
RHS tile ``(32, 128)``). Production writes ``bitcast(u8).reshape(64, 512)`` and
the compiler lowers it to vunpack / vpack / vshrl (3,264 VALU ops per query,
exp-da1zku7f6c). The variants below do the same transpose, one chunk at a time:

* ``A``  production: ``bitcast(u32 -> u8).reshape(64, 512)``.
* ``B``  one ``vperm.slane`` per source vreg (8 tokens) via the
  ``vperm_slane_byte`` idiom of libtpu-agent#292, store to a VMEM scratch,
  then one stride-4 sublane load per target vreg (32 tokens, one column block).
* ``B0`` ``B`` without the vperm: the scratch store + stride load floor. Its
  output is wrong by design (``exact`` must be false).
* ``B2`` registers only: the same vperm, then 4 ``vrot.slane`` + 3 ``vsel`` per
  target vreg (libtpu-agent#297) instead of the scratch round trip.
* ``AR`` registers only, libtpu-agent#309 scheme A: per 32 tokens 4 vperm.slane
  (even / odd patterns) + 4 vsel + 2 vrot.slane + 4 vcombine.{low,high}.

Each variant runs once over a whole query (32 chunks, checked byte-exact against
numpy) and once in a timing loop (KV resident in VMEM, no HBM traffic, so the
time is the reorder itself). Compilation dumps LLO text; ``RSTAT`` lines carry
the opcode histogram of each variant's final bundles. Prints ``R*`` lines.
"""

import collections
import glob
import json
import os
import re
import statistics
import sys
import time
import traceback

DUMP = "/tmp/llo_dump"
os.makedirs(DUMP, exist_ok=True)
os.environ.setdefault("JAX_PLATFORMS", "tpu")
os.environ["LIBTPU_INIT_ARGS"] = (
    os.environ.get("LIBTPU_INIT_ARGS", "")
    + f" --xla_jf_dump_to={DUMP} --xla_jf_dump_llo_text=true"
).strip()

import numpy as np  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402

TOKENS = 2048  # top-k rows per query
CHUNK = 64  # production bkv_sz
NCH = TOKENS // CHUNK
F8 = jnp.float8_e4m3fn


def emit(kind, **kw):
    print(f"{kind} " + json.dumps(kw), flush=True)


# ---------------------------------------------------------------- layouts
def pack_sc(u8):
    """u8[T, 512] token bytes -> u32[T, 128]: word w byte g = byte g*128+w."""
    bands = u8.reshape(u8.shape[0], 4, 128).astype(np.uint32)
    return bands[:, 0] | (bands[:, 1] << 8) | (bands[:, 2] << 16) | (bands[:, 3] << 24)


def swap_pattern():
    """vperm.slane physical control word: out sublane 4h+g byte b <- src sublane 4h+b byte g.

    Physical byte selector = (src_byte << 3) | src_sublane (libtpu-agent#292).
    The 8 source tokens' 4 feature groups become group-major, 4 tokens a word.
    """
    w = np.zeros(8, np.uint32)
    for s in range(8):
        h, g = s >> 2, s & 3
        for b in range(4):
            w[s] |= np.uint32((g << 3) | (4 * h + b)) << np.uint32(8 * b)
    return np.broadcast_to(w[:, None], (8, 128)).copy()


def rel_offsets():
    """Per column block g: out sublane s reads sublane 4*(s&1)+g of its source vreg."""
    s = np.arange(8)
    rel = np.stack([4 * (s & 1) + g - s for g in range(4)]).astype(np.int32)  # [4, 8]
    return np.broadcast_to(rel[:, :, None], (4, 8, 128)).reshape(32, 128).copy()


# ---------------------------------------------------------------- primitives
def vperm_slane_byte(val_u8, pat_u32):
    """One vperm.slane (libtpu-agent#292): pattern holds physical selectors."""
    p_bytes = (pat_u32 & jnp.uint32(0x18181818)) >> jnp.uint32(3)
    p_slanes = (pat_u32 & jnp.uint32(0x07070707)) << jnp.uint32(2)
    logical = pltpu.bitcast(p_bytes | p_slanes, jnp.uint8)
    return jnp.take_along_axis(val_u8, logical, axis=-2, mode="promise_in_bounds",
                               wrap_negative_indices=False)


def vrot_rel(x_u32, rel_i32):
    """One vrot.slane (libtpu-agent#297): out[s] = x[s + rel[s]]."""
    slaneid = lax.broadcasted_iota(jnp.int32, (8, 128), 0)
    return jnp.take_along_axis(x_u32, slaneid + rel_i32, axis=0, mode="promise_in_bounds",
                               wrap_negative_indices=False)


def stride4_rows(ref, start):
    """8 sublanes start, start+4, ..., start+28 of a u32 VMEM ref."""
    try:
        return ref[pl.ds(start, 8, stride=4), :]
    except TypeError:
        return ref[pl.Slice(start, 8, 4), :]


# ---------------------------------------------------------------- variants
# Each writes chunk `row0` of src_ref into out_ref rows [dst, dst+64) and uses
# scratch slot `slot` (static) so consecutive chunks do not serialize on it.
def var_a(src_ref, row0, out_ref, dst, ctx, slot):
    x = src_ref[pl.ds(row0, CHUNK), :]
    kv = pltpu.bitcast(pltpu.bitcast(x, jnp.uint8).reshape(CHUNK, 512), F8)
    out_ref[pl.ds(dst, CHUNK), :] = kv


def _var_b(src_ref, row0, out_ref, dst, ctx, slot, use_vperm):
    scr = ctx["scratch"]
    base = slot * CHUNK
    for k in range(CHUNK // 8):
        x = src_ref[pl.ds(row0 + 8 * k, 8), :]
        if use_vperm:
            x = pltpu.bitcast(vperm_slane_byte(pltpu.bitcast(x, jnp.uint8), ctx["pat"]),
                              jnp.uint32)
        scr[pl.ds(base + 8 * k, 8), :] = x
    for m in range(CHUNK // 32):
        for g in range(4):
            t = stride4_rows(scr, base + 32 * m + g)
            out_ref[pl.ds(dst + 32 * m, 32), pl.ds(128 * g, 128)] = pltpu.bitcast(t, F8)


def var_b(*a):
    _var_b(*a, use_vperm=True)


def var_b0(*a):
    _var_b(*a, use_vperm=False)


def var_b2(src_ref, row0, out_ref, dst, ctx, slot):
    slaneid = lax.broadcasted_iota(jnp.int32, (8, 128), 0)
    pair = slaneid >> 1
    for m in range(CHUNK // 32):
        perm = []
        for k in range(4):
            x = src_ref[pl.ds(row0 + 32 * m + 8 * k, 8), :]
            perm.append(pltpu.bitcast(
                vperm_slane_byte(pltpu.bitcast(x, jnp.uint8), ctx["pat"]), jnp.uint32))
        for g in range(4):
            rel = ctx["rel"][g]
            t = vrot_rel(perm[0], rel)
            for k in range(1, 4):
                t = jnp.where(pair == k, vrot_rel(perm[k], rel), t)
            out_ref[pl.ds(dst + 32 * m, 32), pl.ds(128 * g, 128)] = pltpu.bitcast(t, F8)


def vcombine_low_high(x, y):
    """vcombine.low / vcombine.high of two u32[8, 128] (libtpu-agent#309 idiom):
    rows 0..3 of x then 0..3 of y, and rows 4..7 of x then 4..7 of y."""
    x32 = pltpu.bitcast(x, jnp.int16).astype(jnp.int32)
    y32 = pltpu.bitcast(y, jnp.int16).astype(jnp.int32)
    lo = pltpu.bitcast(jnp.concatenate([x32[:8], y32[:8]], axis=0).astype(jnp.int16), jnp.uint32)
    hi = pltpu.bitcast(jnp.concatenate([x32[8:], y32[8:]], axis=0).astype(jnp.int16), jnp.uint32)
    return lo, hi


def scheme_a_patterns():
    """Built from iota at kernel top and closed over (never a loop carry), so the
    #292 bit-swap folds away. Even source vregs: out sublane 2g+h byte b <- src
    sublane 4h+b byte g; odd ones: the same rotated down two sublanes."""
    slane = lax.broadcasted_iota(jnp.uint32, (8, 128), 0)
    even = (slane << 2) * jnp.uint32(0x01010101) + jnp.uint32(0x03020100)
    odd = (((slane - jnp.uint32(2)) & jnp.uint32(7)) << 2) * jnp.uint32(0x01010101) \
        + jnp.uint32(0x03020100)
    return even, odd, (slane & jnp.uint32(2)) != jnp.uint32(0)


def scheme_a_32(src_ref, row0, pats):
    """u32 rows [row0, row0+32) -> 4 u32[8, 128] column-block vregs (fp8 tiles)."""
    even, odd, sel = pats
    p = [pltpu.bitcast(vperm_slane_byte(
        pltpu.bitcast(src_ref[pl.ds(row0 + 8 * k, 8), :], jnp.uint8),
        even if k % 2 == 0 else odd), jnp.uint32) for k in range(4)]
    a01 = jnp.where(sel, p[1], p[0])
    b01 = pltpu.roll(jnp.where(sel, p[0], p[1]), 6, axis=0)  # out[s] = x[s + 2]
    a23 = jnp.where(sel, p[3], p[2])
    b23 = pltpu.roll(jnp.where(sel, p[2], p[3]), 6, axis=0)
    t0, t2 = vcombine_low_high(a01, a23)
    t1, t3 = vcombine_low_high(b01, b23)
    return t0, t1, t2, t3


def var_ar(src_ref, row0, out_ref, dst, ctx, slot):
    for m in range(CHUNK // 32):
        for g, t in enumerate(scheme_a_32(src_ref, row0 + 32 * m, ctx["pats"])):
            out_ref[pl.ds(dst + 32 * m, 32), pl.ds(128 * g, 128)] = pltpu.bitcast(t, F8)


VARIANTS = {"A": var_a, "B": var_b, "B0": var_b0, "B2": var_b2, "AR": var_ar}
NEEDS_SCRATCH = {"B", "B0"}


def build(name, n_pairs, out_rows, steps, check):
    """n_pairs loop iterations, two chunks each (static scratch slots 0 / 1)."""
    fn = VARIANTS[name]

    def kernel(src_ref, pat_ref, rel_ref, out_ref, *scratch):
        ctx = {"pat": pat_ref[...], "pats": scheme_a_patterns() if name == "AR" else None,
               "rel": [rel_ref[pl.ds(8 * g, 8), :] for g in range(4)],
               "scratch": scratch[0] if scratch else None}

        def body(j, carry):
            for half in range(2):
                i = 2 * j + half
                row0 = pl.multiple_of(lax.rem(i, NCH) * CHUNK, CHUNK)
                dst = pl.multiple_of(i * CHUNK, CHUNK) if check else half * CHUNK
                fn(src_ref, row0, out_ref, dst, ctx, half)
            return carry

        lax.fori_loop(0, n_pairs, body, 0)

    scratch_shapes = [pltpu.VMEM((2 * CHUNK, 128), jnp.uint32)] if name in NEEDS_SCRATCH else []
    full = lambda shape: pl.BlockSpec(shape, lambda s: (0, 0))
    call = pl.pallas_call(
        kernel,
        grid=(steps,),
        in_specs=[full((TOKENS, 128)), full((8, 128)), full((32, 128))],
        out_specs=full((out_rows, 512)),
        out_shape=jax.ShapeDtypeStruct((out_rows, 512), F8),
        scratch_shapes=scratch_shapes,
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024 * 1024,
                                             dimension_semantics=("arbitrary",)),
        name=f"reorder_{name}_{'check' if check else 'time'}",
    )
    return jax.jit(call)


# ---------------------------------------------------------------- LLO stats
BUNDLE_RE = re.compile(r"\s*(0x[0-9a-f]+)\s*:\s*(>*)\s*\{(.*)\}\s*$")
TOK_RE = re.compile(r"(?<![%\w.$#])([a-z][a-z0-9_]*(?:\.[\w]+)*)")
KEEP = ("vperm", "vrot", "vld", "vst", "vunpack", "vpack", "vshrl", "vshll", "vshr", "vshl",
        "vsel", "vand", "vor", "vcombine", "vsub", "vadd", "vmov", "vlaneseq", "vxpose")


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


def llo_report(tag, new_files):
    names = sorted(f for f in new_files if f.endswith("final_bundles.txt") and "reorder_" in f)
    if not names:
        emit("RLLOFILES", tag=tag, files=sorted(os.path.basename(f)[-80:] for f in new_files)[:30])
    for f in names:
        rows = []
        for ln in open(f, errors="replace"):
            m = BUNDLE_RE.match(ln)
            if m:
                rows.append(opcodes(m.group(3)))
        hist = collections.Counter(o for ops in rows for o in ops)
        keep = {o: n for o, n in sorted(hist.items()) if o.startswith(KEEP)}
        valu = sum(n for o, n in hist.items() if o.startswith("v") and not o.startswith(
            ("vld", "vst", "vmat", "vpop", "vdelay", "vnop", "vwait", "vsync")))
        emit("RSTAT", tag=tag, file=os.path.basename(f)[-80:], bundles=len(rows),
             vector_non_mem_ops=valu, ops=keep)
        raw = [re.sub(r"/\*.*?\*/", "", ln).strip()[:300] for ln in open(f, errors="replace")
               if re.search(r"vperm|vrot|vld\.|vld |vunpack|vpack", ln)][:24]
        emit("RRAW", tag=tag, lines=raw)


# ---------------------------------------------------------------- main
def main():
    try:
        from importlib.metadata import version
        libtpu = version("libtpu")
    except Exception as e:  # noqa: BLE001
        libtpu = repr(e)
    dev = jax.devices()[0]
    emit("RINFO", jax=jax.__version__, libtpu=libtpu, device=dev.device_kind)

    rng = np.random.default_rng(0)
    tok = rng.integers(0, 256, size=(TOKENS, 512), dtype=np.uint8)
    src = jnp.asarray(pack_sc(tok))
    pat = jnp.asarray(swap_pattern())
    rel = jnp.asarray(rel_offsets())

    steps = int(os.environ.get("R_STEPS", 8))
    nrep = int(os.environ.get("R_NREP", 4096))  # chunks per grid step
    names = sys.argv[1:] or list(VARIANTS)
    for name in names:
        row = {"variant": name}
        try:
            before = files_now()
            f = build(name, NCH // 2, TOKENS, 1, check=True)
            out = np.asarray(jax.block_until_ready(f(src, pat, rel))).view(np.uint8)
            row["exact"] = bool(np.array_equal(out, tok))
            if not row["exact"]:
                bad = np.argwhere(out != tok)
                row["mismatch"] = int(bad.shape[0])
                row["first_bad"] = bad[:4].tolist()
            llo_report(f"{name}_check", files_now() - before)

            before = files_now()
            f = build(name, nrep // 2, 2 * CHUNK, steps, check=False)
            jax.block_until_ready(f(src, pat, rel))
            llo_report(f"{name}_time", files_now() - before)
            runs = []
            for _ in range(7):
                t0 = time.perf_counter()
                jax.block_until_ready(f(src, pat, rel))
                runs.append(time.perf_counter() - t0)
            med = statistics.median(runs)
            chunks = steps * nrep
            row["ms"] = round(med * 1e3, 3)
            row["us_per_chunk"] = round(med * 1e6 / chunks, 4)
            row["us_per_query"] = round(med * 1e6 / chunks * NCH, 3)
            row["ns_range"] = [round(min(runs) * 1e9 / chunks, 1), round(max(runs) * 1e9 / chunks, 1)]
        except Exception as e:  # noqa: BLE001 - report, keep the other variants
            traceback.print_exc()
            row["fail"] = repr(e)[:1500]
        emit("RRES", **row)
    print("R_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
