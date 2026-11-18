"""Test only: can contiguous runs of top-k rows make the extend gather cheaper?

Two parts, one pod:

1. ``RUN`` (offline, the real GLM-5.2 top-k dump from exp-fj8ahrq5pz): per query,
   sort its 2048 logical positions and count
   * maximal runs of consecutive positions, split at page (64) boundaries since
     pages are not physically adjacent;
   * aligned blocks of B = 8 / 16 / 32 / 64 rows touched (one DMA each if a
     block is fetched whole), and the over-read (block rows / 2048);
   * the same blocks unioned over a 128-query chunk (page-granular fetch for
     the chunk, a per-query mask then restores the exact top-k).
2. ``SEG`` (TPU): what a DMA of L contiguous rows costs. A TensorCore Pallas
   kernel copies ``nseg`` segments of L NoPE rows (u32[L, 128], 512 B/row) from
   random starts HBM -> HBM, ``depth`` copies in flight. L = 1 .. 64 at a fixed
   total of 262,144 rows (one 128-query chunk's gather). Also the native SC
   gather on the same total with random rows and with rows in runs of L
   (``SCRUN``), which shows whether the existing gather gains from contiguity.

Prints ``RUN`` / ``SEG`` / ``SCRUN`` JSON lines.
"""

import glob
import json
import os
import statistics
import sys
import time

import numpy as np

PAGE = 64
TOPK = 2048
CHUNK = 128
CACHE_ROWS = 51264
TOTAL = CHUNK * TOPK


def run_stats(raw_dir):
    files = sorted(glob.glob(f"{raw_dir}/**/raw-*.npy", recursive=True))
    for f in files:
        r = np.load(f).astype(np.int64)
        rows = [np.sort(x[x >= 0]) for x in r]
        runs = []
        for x in rows:
            brk = (np.diff(x) != 1) | ((x[1:] % PAGE) == 0)
            runs.append(1 + int(brk.sum()))
        runs = np.array(runs)
        out = {"f": os.path.basename(f), "queries": len(rows),
               "runs_mean": round(float(runs.mean()), 1),
               "runs_p10_p90": [int(np.percentile(runs, 10)), int(np.percentile(runs, 90))],
               "rows_per_run": round(TOPK / float(runs.mean()), 2)}
        for b in (8, 16, 32, 64):
            per_q = np.array([np.unique(x // b).size for x in rows])
            chunk = [np.unique(np.concatenate(rows[c:c + CHUNK]) // b).size
                     for c in range(0, len(rows) - CHUNK + 1, CHUNK)]
            out[f"b{b}"] = {
                "blocks_q": round(float(per_q.mean()), 1),
                "overread_q": round(float(per_q.mean()) * b / TOPK, 2),
                "blocks_chunk": round(float(np.mean(chunk)), 1),
                # rows fetched per chunk vs per-query row gather (CHUNK * TOPK)
                "rows_chunk_frac": round(float(np.mean(chunk)) * b / TOTAL, 3),
            }
        print("RUN " + json.dumps(out), flush=True)


def timed(fn, args, iters=20, repeats=5):
    import jax

    jax.block_until_ready(fn(*args))
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            o = fn(*args)
        jax.block_until_ready(o)
        runs.append((time.perf_counter() - t0) / iters * 1e6)
    return round(statistics.median(runs), 1), [round(min(runs), 1), round(max(runs), 1)]


def seg_kernel(L, nseg, depth):
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    # Starts go through SMEM in blocks of G per grid step (all of them at once
    # would not fit in SMEM at L = 1).
    G = min(nseg, 1024)
    nblk = nseg // G

    def body(starts_ref, src, dst, sems):
        b = pl.program_id(0)

        def cp(i):
            return pltpu.make_async_copy(src.at[pl.ds(starts_ref[0, i], L)],
                                         dst.at[pl.ds((b * G + i) * L, L)],
                                         sems.at[i % depth])

        def step(i, c):
            @pl.when(i >= depth)
            def _():
                cp(i - depth).wait()

            cp(i).start()
            return c

        lax.fori_loop(0, G, step, 0)

        def drain(i, c):
            cp(i).wait()
            return c

        lax.fori_loop(max(0, G - depth), G, drain, 0)

    return jax.jit(lambda starts, src: pl.pallas_call(
        body,
        grid=(nblk,),
        in_specs=[pl.BlockSpec((1, G), lambda b: (b, 0), memory_space=pltpu.SMEM),
                  pl.BlockSpec(memory_space=pltpu.HBM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=[pltpu.SemaphoreType.DMA((depth,))],
        out_shape=jax.ShapeDtypeStruct((nseg * L, 128), jnp.uint32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name=f"seg_L{L}_d{depth}",
    )(starts.reshape(nblk, G), src))


def seg_bench():
    import jax.numpy as jnp
    from sglang_tpu.attention.pallas.mla_sparse import native_sc_gather as nsc

    rng = np.random.default_rng(0)
    nope = jnp.asarray(rng.integers(0, 2**32, (CACHE_ROWS, 128), dtype=np.uint32))
    rope = jnp.asarray(rng.integers(0, 2**32, (CACHE_ROWS // 4, 128), dtype=np.uint32))
    for L in (1, 2, 4, 8, 16, 32, 64):
        nseg = TOTAL // L
        starts = jnp.asarray((rng.integers(0, CACHE_ROWS // L, nseg) * L).astype(np.int32))
        for depth in (16, 64):
            row = {"L": L, "nseg": nseg, "depth": depth}
            try:
                f = seg_kernel(L, nseg, depth)
                out = f(starts, nope)
                s0 = int(starts[3])
                row["ok"] = bool(jnp.array_equal(out[3 * L:4 * L], nope[s0:s0 + L]))
                us, rr = timed(f, (starts, nope))
                row.update(us=us, us_range=rr, ns_per_row=round(us * 1e3 / TOTAL, 3),
                           ns_per_seg=round(us * 1e3 / nseg, 1))
            except Exception as e:  # report, do not hide
                row["fail"] = repr(e)[:400]
            print("SEG " + json.dumps(row), flush=True)
    import jax

    for L in (1, 4, 16, 64):
        nseg = TOTAL // L
        st = rng.integers(0, CACHE_ROWS // L, nseg) * L
        ix = (st[:, None] + np.arange(L)[None]).reshape(CHUNK, TOPK).astype(np.int32)
        row = {"L": L}
        try:
            g = jax.jit(lambda n, r, i: nsc.dsa_gather_native_sc(
                n, r, i, out_size=TOTAL, atoms_per_batch=16))
            us, rr = timed(g, (nope, rope, jnp.asarray(ix)))
            row.update(us=us, us_range=rr, ns_per_row=round(us * 1e3 / TOTAL, 3))
        except Exception as e:  # report, do not hide
            row["fail"] = repr(e)[:400]
        print("SCRUN " + json.dumps(row), flush=True)


def main():
    raw = sys.argv[1] if len(sys.argv) > 1 else "/old/fj8"
    try:
        run_stats(raw)
    except Exception as e:  # report, do not hide
        print("RUN " + json.dumps({"fail": repr(e)[:400]}), flush=True)
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    seg_bench()
    print("GR_DONE", flush=True)


if __name__ == "__main__":
    main()
