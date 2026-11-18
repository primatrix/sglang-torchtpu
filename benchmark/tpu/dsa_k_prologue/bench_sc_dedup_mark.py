"""Test only: dedup one gather chunk's top-k rows on the SparseCore with a bitmap.

The extend gather reads 128 queries x 2048 rows per chunk, and the GLM-5.2
top-k (exp-fj8ahrq5pz) repeats rows heavily: only ~3-5% of a chunk's rows are
distinct. Gathering only those takes ~65 us instead of ~185 us, but XLA's
``unique`` takes 2.3 ms per chunk (it sorts 262,144 values). This kernel does
the dedup on the SparseCore instead, one chunk per vector subcore, in three
passes over a bitmap of the cache rows (``P`` bits, in TileSPMEM):

1. mark: for each top-k row (no repeats inside a row) set its position's bit;
   a lane only adds its bit if the bit is not set yet, so adds are ORs.
2. emit: per 16 bitmap words, prefix popcount -> ``prefix[word]``; then scan
   positions in order and compress the set ones: ``distinct`` comes out sorted.
3. inverse: for each top-k entry, rank = prefix[word] + popcount(lower bits).
   ``distinct[inverse] == topk`` for valid entries, ``-1`` stays ``-1``.

Prints one ``DEDUP`` JSON line per point (``part: sc_mark``) with the time for
``--chunks`` chunks in one call and whether the output matches numpy.
"""

import argparse
import functools
import json
import os
import statistics
import time

os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.experimental.pallas import tpu_sc as plsc  # noqa: E402


def _popcount(x):
    """Per-lane bit count of an int32 vector. ``lax.population_count`` has no
    SparseCore lowering; shifts, ands and adds do (no multiply)."""
    x = x.astype(jnp.int32)
    srl = lax.shift_right_logical
    x = x - jnp.bitwise_and(srl(x, 1), jnp.int32(0x55555555))
    x = jnp.bitwise_and(x, jnp.int32(0x33333333)) + jnp.bitwise_and(srl(x, 2), jnp.int32(0x33333333))
    x = jnp.bitwise_and(x + srl(x, 4), jnp.int32(0x0F0F0F0F))
    x = x + srl(x, 8)
    x = x + srl(x, 16)
    return jnp.bitwise_and(x, jnp.int32(0x3F))

LANES = 16
TOPK = 2048
QUERIES = 128  # rows per chunk = the gather chunk
ROWS = 2  # top-k rows per HBM copy
BLOCK = 1024  # positions per emit block (one HBM write each)


def _kernel(topk_hbm, distinct_hbm, count_hbm, inv_hbm,
            bitmap, prefix, rowbuf, invbuf, outbuf, cntbuf, *, chunks, p_pad):
    sc = pltpu.get_tpu_info().sparse_core
    c = lax.axis_index("core") * sc.num_subcores + lax.axis_index("subcore")
    nwords = p_pad // 32
    iota = lax.iota(jnp.int32, LANES)
    one = jnp.int32(1)

    @pl.when(c < chunks)
    def _():
        def zero(i):
            bitmap[pl.ds(i * LANES, LANES)] = jnp.zeros((LANES,), jnp.int32)

        plsc.parallel_loop(0, nwords // LANES, unroll=8)(zero)

        def rows_in(r):
            pltpu.sync_copy(
                topk_hbm.at[pl.ds((c * QUERIES + r * ROWS) * TOPK, ROWS * TOPK)], rowbuf)

        # 1. mark
        def mark_block(r, carry):
            rows_in(r)
            for k in range(ROWS):
                def mark(i, k=k):
                    p = rowbuf[pl.ds(k * TOPK + i * LANES, LANES)]
                    ok = p >= 0
                    p = jnp.maximum(p, 0)
                    w = jnp.right_shift(p, 5)
                    b = jnp.bitwise_and(p, 31)
                    old = plsc.load_gather(bitmap, [w])
                    new = ok & (jnp.bitwise_and(jnp.right_shift(old, b), one) == 0)
                    plsc.addupdate_scatter(bitmap, (w,), jnp.left_shift(one, b), mask=new)

                plsc.parallel_loop(0, TOPK // LANES, unroll=4)(mark)
            return carry

        lax.fori_loop(0, QUERIES // ROWS, mark_block, jnp.int32(0))

        # 2a. prefix popcount per word
        def pre(i, carry):
            cnt = _popcount(bitmap[pl.ds(i * LANES, LANES)])
            incl = plsc.cumsum(cnt) + carry
            prefix[pl.ds(i * LANES, LANES)] = incl - cnt
            return incl[LANES - 1]

        plsc.parallel_loop(0, nwords // LANES, unroll=2, carry=jnp.int32(0))(pre)

        # 2b. emit sorted distinct positions. HBM DMA offsets must be word
        # aligned, so outbuf carries a partial block across positions and only
        # whole BLOCKs are written, at multiples of BLOCK.
        base = c * (p_pad + BLOCK)

        def emit_block(blk, carry):
            woff, flushed = carry

            def emit(i, woff):
                p = blk * BLOCK + i * LANES + iota
                bits = plsc.load_gather(bitmap, [jnp.right_shift(p, 5)])
                sel = jnp.bitwise_and(jnp.right_shift(bits, jnp.bitwise_and(p, 31)), one) == 1
                plsc.store_compressed(outbuf.at[pl.ds(woff, LANES)], p, mask=sel)
                return woff + plsc.all_reduce_population_count(sel)[0]

            woff = plsc.parallel_loop(0, BLOCK // LANES, unroll=4, carry=woff)(emit)
            full = woff >= BLOCK

            @pl.when(full)
            def _():
                pltpu.sync_copy(outbuf.at[pl.ds(0, BLOCK)],
                                distinct_hbm.at[pl.ds(base + flushed, BLOCK)])
                for i in range(BLOCK // LANES):
                    outbuf[pl.ds(i * LANES, LANES)] = outbuf[pl.ds(BLOCK + i * LANES, LANES)]

            return (jnp.where(full, woff - BLOCK, woff),
                    jnp.where(full, flushed + BLOCK, flushed))

        woff, flushed = lax.fori_loop(0, p_pad // BLOCK, emit_block,
                                      (jnp.int32(0), jnp.int32(0)))
        # Last partial block (entries past woff are padding, beyond count).
        pltpu.sync_copy(outbuf.at[pl.ds(0, BLOCK)],
                        distinct_hbm.at[pl.ds(base + flushed, BLOCK)])
        total = flushed + woff
        cntbuf[...] = jnp.full((LANES,), total, jnp.int32)
        pltpu.sync_copy(cntbuf, count_hbm.at[pl.ds(c * LANES, LANES)])

        # 3. inverse index
        def inv_block(r, carry):
            rows_in(r)

            def inv(i):
                p = rowbuf[pl.ds(i * LANES, LANES)]
                ok = p >= 0
                p = jnp.maximum(p, 0)
                w = jnp.right_shift(p, 5)
                b = jnp.bitwise_and(p, 31)
                word = plsc.load_gather(bitmap, [w])
                below = jnp.bitwise_and(word, jnp.left_shift(one, b) - 1)
                rank = plsc.load_gather(prefix, [w]) + _popcount(below)
                invbuf[pl.ds(i * LANES, LANES)] = jnp.where(ok, rank, -1)

            plsc.parallel_loop(0, ROWS * TOPK // LANES, unroll=4)(inv)
            pltpu.sync_copy(
                invbuf, inv_hbm.at[pl.ds((c * QUERIES + r * ROWS) * TOPK, ROWS * TOPK)])
            return carry

        lax.fori_loop(0, QUERIES // ROWS, inv_block, jnp.int32(0))


@functools.partial(jax.jit, static_argnames=("p_pad",))
def sc_dedup(topk, *, p_pad):
    """``i32[chunks * 128, 2048]`` rows (< p_pad, -1 padded, no repeats per row)
    -> (distinct ``i32[chunks, p_pad + 1024]`` sorted, valid up to count;
    count ``i32[chunks]``; inverse ``i32[chunks * 128, 2048]``)."""
    chunks = topk.shape[0] // QUERIES
    sc = pltpu.get_tpu_info().sparse_core
    mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore",
                                  num_cores=sc.num_cores, num_subcores=sc.num_subcores)
    distinct, count, inv = pl.kernel(
        functools.partial(_kernel, chunks=chunks, p_pad=p_pad),
        out_type=(jax.ShapeDtypeStruct((chunks * (p_pad + BLOCK),), jnp.int32),
                  jax.ShapeDtypeStruct((chunks * LANES,), jnp.int32),
                  jax.ShapeDtypeStruct((topk.size,), jnp.int32)),
        mesh=mesh,
        scratch_types=(
            pltpu.VMEM((p_pad // 32,), jnp.int32),  # bitmap
            pltpu.VMEM((p_pad // 32,), jnp.int32),  # prefix
            pltpu.VMEM((ROWS * TOPK,), jnp.int32),  # rowbuf
            pltpu.VMEM((ROWS * TOPK,), jnp.int32),  # invbuf
            pltpu.VMEM((2 * BLOCK + LANES,), jnp.int32),  # outbuf (carry-over)
            pltpu.VMEM((LANES,), jnp.int32),  # cntbuf
        ),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True,
                                             needs_layout_passes=False),
        name=f"sc_dedup_mark_c{chunks}_p{p_pad}",
    )(topk.reshape(-1))
    return (distinct.reshape(chunks, p_pad + BLOCK), count.reshape(chunks, LANES)[:, 0],
            inv.reshape(topk.shape))


def make_topk(rng, chunks, pool, cache_rows):
    out = np.full((chunks * QUERIES, TOPK), -1, np.int32)
    for c in range(chunks):
        pool_rows = rng.choice(cache_rows, pool, replace=False)
        for q in range(QUERIES):
            n = TOPK - int(rng.integers(0, 64))  # a few -1 pads per row
            out[c * QUERIES + q, :n] = rng.choice(pool_rows, n, replace=False)
    return out


def check(topk, distinct, count, inv):
    ok = True
    for c in range(topk.shape[0] // QUERIES):
        blk = topk[c * QUERIES:(c + 1) * QUERIES]
        ref = np.unique(blk[blk >= 0])
        n = int(count[c])
        if n != ref.size or not np.array_equal(distinct[c, :n], ref):
            return False
        iv = inv[c * QUERIES:(c + 1) * QUERIES]
        ok &= bool(np.array_equal(iv < 0, blk < 0))
        ok &= bool(np.array_equal(distinct[c][np.maximum(iv, 0)][blk >= 0], blk[blk >= 0]))
    return ok


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--cache-rows", type=int, default=51264)  # extend_c1 flat slots
    a.add_argument("--iters", type=int, default=20)
    a.add_argument("--repeats", type=int, default=5)
    a = a.parse_args()
    p_pad = -(-a.cache_rows // BLOCK) * BLOCK
    rng = np.random.default_rng(0)
    for chunks in (1, 16):
        for pool in (8192, 12288, 16384):
            row = dict(part="sc_mark", chunks=chunks, pool=pool, cache_rows=a.cache_rows)
            try:
                topk = make_topk(rng, chunks, pool, a.cache_rows)
                x = jnp.asarray(topk)
                out = jax.block_until_ready(sc_dedup(x, p_pad=p_pad))
                d, n, iv = (np.asarray(o) for o in out)
                row["exact"] = check(topk, d, n, iv)
                row["distinct"] = int(n[0])
                runs = []
                for _ in range(a.repeats):
                    t0 = time.perf_counter()
                    for _ in range(a.iters):
                        out = sc_dedup(x, p_pad=p_pad)
                    jax.block_until_ready(out)
                    runs.append((time.perf_counter() - t0) / a.iters * 1e6)
                row["us"] = round(statistics.median(runs), 1)
                row["us_range"] = [round(min(runs), 1), round(max(runs), 1)]
                row["us_per_chunk_parallel"] = row["us"]  # chunks run on separate subcores
            except Exception as e:  # report, do not hide
                row["error"] = repr(e)[:1500]
            print("DEDUP " + json.dumps(row), flush=True)
    print("DEDUP_DONE", flush=True)


if __name__ == "__main__":
    main()
