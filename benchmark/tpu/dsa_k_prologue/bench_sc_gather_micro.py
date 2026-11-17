"""Where the native SparseCore gather spends its time (extend, PR 2198 shapes).

The extend step is ~3.1 ms of SparseCore gather with the TensorCore idle, and
packing RoPE from 128 to 64 B/token (sgl-project/sglang-torchtpu#2212, -10%
bytes) only saved 1-3%. So bytes are not the limit. This times one chunk's
gather (128 queries x 2048 rows, the per-call size of the extend step) with
everything but one factor fixed:

* ``pattern``: the row indices. ``random`` (the extend step's access), ``sorted``
  (same rows, ascending per query), ``contiguous`` (consecutive rows), ``same``
  (one row repeated). Random ~= contiguous means the limit is per-row work
  (descriptor issue / DMA count), not HBM locality.
* ``cache_rows``: 51,264 (extend, 33 MB NoPE) vs 360,768 (verify, 185 MB).
* ``atoms`` / ``wait_mode``: the kernel's pipelining knobs.

Plus two references on the same bytes: an XLA ``take`` (TensorCore gather) and a
dense copy (HBM streaming). Prints one ``MICRO`` JSON line per point: us per
call, rows/us, GB/s of NoPE + RoPE bytes moved.
"""

import argparse
import json
import os
import statistics
import time

os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from sglang_tpu.attention.pallas.mla_sparse import native_sc_gather as nsc  # noqa: E402

TOPK = 2048
ROW_BYTES = 512 + 128  # NoPE + RoPE per gathered row, as the cache stores them


def indices(pattern, queries, cache_rows, rng):
    n = queries * TOPK
    if pattern == "random":
        # Per query: TOPK distinct rows of the cache, as the indexer picks them.
        return np.stack([rng.choice(cache_rows, TOPK, replace=False)
                         for _ in range(queries)]).astype(np.int32)
    if pattern == "sorted":
        return np.sort(indices("random", queries, cache_rows, rng), axis=1)
    if pattern == "contiguous":
        return (np.arange(n, dtype=np.int64) % cache_rows).astype(np.int32).reshape(queries, TOPK)
    if pattern == "same":
        return np.full((queries, TOPK), 37, np.int32)
    raise ValueError(pattern)


def timed(fn, args, iters, repeats):
    out = fn(*args)
    jax.block_until_ready(out)
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn(*args)
        jax.block_until_ready(out)
        runs.append((time.perf_counter() - t0) / iters * 1e6)
    return statistics.median(runs), [round(min(runs), 1), round(max(runs), 1)]


def emit(row, us, rng_us, rows):
    row.update(us=round(us, 1), us_range=rng_us, rows_per_us=round(rows / us, 1),
               gbps=round(rows * ROW_BYTES / us / 1e3, 1))
    print("MICRO " + json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", type=int, default=128)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--repeats", type=int, default=5)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    rows = a.queries * TOPK
    for cache_rows in (51264, 360768):
        nope = jnp.asarray(rng.integers(0, 2**32, (cache_rows, 128), dtype=np.uint32))
        rope = jnp.asarray(rng.integers(0, 2**32, (cache_rows // 4, 128), dtype=np.uint32))
        idx = {pt: jnp.asarray(indices(pt, a.queries, cache_rows, rng))
               for pt in ("random", "sorted", "contiguous", "same")}
        for pattern, ix in idx.items():
            for atoms in (4, 8, 16):
                for wait in ("overlap_rope", "before_rope"):
                    if wait == "before_rope" and (pattern != "random" or atoms != 16):
                        continue
                    f = jax.jit(lambda n, r, i, _a=atoms, _w=wait: nsc.dsa_gather_native_sc(
                        n, r, i, out_size=rows, atoms_per_batch=_a, wait_mode=_w))
                    row = dict(kind="sc_gather", cache_rows=cache_rows, pattern=pattern,
                               atoms=atoms, wait=wait)
                    try:
                        us, rr = timed(f, (nope, rope, ix), a.iters, a.repeats)
                        emit(row, us, rr, rows)
                    except Exception as e:  # report, do not hide
                        row["error"] = repr(e)[:500]
                        print("MICRO " + json.dumps(row), flush=True)
        # References on the same bytes: TensorCore XLA gather, and a dense copy.
        rope_rows = rope.reshape(-1, 32)
        take = jax.jit(lambda n, r, i: (jnp.take(n, i.reshape(-1), axis=0),
                                         jnp.take(r, i.reshape(-1), axis=0)))
        for pattern in ("random", "contiguous"):
            us, rr = timed(take, (nope, rope_rows, idx[pattern]), a.iters, a.repeats)
            emit(dict(kind="xla_take", cache_rows=cache_rows, pattern=pattern), us, rr, rows)
        n_src = nope[: min(rows, cache_rows)]
        reps = -(-rows // n_src.shape[0])
        copy = jax.jit(lambda n: jnp.tile(n, (reps, 1))[:rows] + jnp.uint32(1))
        us, rr = timed(copy, (n_src,), a.iters, a.repeats)
        # Dense: NoPE bytes only; scale to ROW_BYTES for the same gbps column.
        emit(dict(kind="dense_copy_nope_only", cache_rows=cache_rows), us * ROW_BYTES / 512,
             rr, rows)
    print("MICRO_DONE", flush=True)


if __name__ == "__main__":
    main()
