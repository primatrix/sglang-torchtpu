"""Where the native SparseCore gather spends its time (extend, PR 2198 shapes).

The extend step is ~3.1 ms of SparseCore gather with the TensorCore idle, and
packing RoPE from 128 to 64 B/token (sgl-project/sglang-torchtpu#2212, -10%
bytes) saved only 1-3%. The first sweep (exp-i0f2a5g6yr) found random, sorted
and contiguous indices equally fast and a 57x slowdown when every index is the
same row. This sweep separates per-row cost from per-byte cost on one chunk's
gather (``queries`` x 2048 rows), one factor at a time:

* ``rows``: queries 32 / 64 / 128 / 256 at fixed bytes per row. Time linear in
  rows with no intercept means no fixed per-call cost worth chasing.
* ``bytes``: RoPE 128 B (32 words) vs 64 B (16 words) per row at fixed rows,
  the only byte knob the kernel has (NoPE is always 512 B). Gives the per-byte
  slope; with the rows sweep, the per-row slope.
* ``atoms``: 4 .. 64 owner-batch atoms per DMA round.
* ``shared``: every query gathers the same 2048 rows (permuted), the far end of
  the cross-query reuse extend top-k has; ``same`` is one row everywhere.

Reference: a dense elementwise copy of the same bytes (HBM streaming). Prints
one ``MICRO`` JSON line per point. Needs the #2212 gather (rope_words derived
from the cache shape).
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
CACHE_ROWS = 51264  # extend_c1's 801 pages x 64


def indices(pattern, queries, rng):
    if pattern == "random":
        # Per query: TOPK distinct rows, as the indexer picks them.
        return np.stack([rng.choice(CACHE_ROWS, TOPK, replace=False)
                         for _ in range(queries)]).astype(np.int32)
    if pattern == "shared":
        base = rng.choice(CACHE_ROWS, TOPK, replace=False)
        return np.stack([rng.permutation(base) for _ in range(queries)]).astype(np.int32)
    if pattern == "same":
        return np.full((queries, TOPK), 37, np.int32)
    raise ValueError(pattern)


def timed(fn, args, iters, repeats):
    jax.block_until_ready(fn(*args))
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn(*args)
        jax.block_until_ready(out)
        runs.append((time.perf_counter() - t0) / iters * 1e6)
    return statistics.median(runs), [round(min(runs), 1), round(max(runs), 1)]


def caches(rope_words, rng):
    nope = jnp.asarray(rng.integers(0, 2**32, (CACHE_ROWS, 128), dtype=np.uint32))
    tokens_per_row = 128 // rope_words
    rope = jnp.asarray(rng.integers(
        0, 2**32, (CACHE_ROWS // tokens_per_row, 128), dtype=np.uint32))
    return nope, rope


def point(a, nope, rope, ix, *, sweep, queries, rope_words, atoms, pattern):
    rows = queries * TOPK
    row_bytes = 512 + 4 * rope_words
    row = dict(sweep=sweep, queries=queries, rows=rows, rope_bytes=4 * rope_words,
               atoms=atoms, pattern=pattern)
    f = jax.jit(lambda n, r, i: nsc.dsa_gather_native_sc(
        n, r, i, out_size=rows, atoms_per_batch=atoms))
    try:
        us, rr = timed(f, (nope, rope, ix), a.iters, a.repeats)
        row.update(us=round(us, 1), us_range=rr, ns_per_row=round(us * 1e3 / rows, 3),
                   gbps=round(rows * row_bytes / us / 1e3, 1))
    except Exception as e:  # report, do not hide
        row["error"] = repr(e)[:500]
    print("MICRO " + json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--repeats", type=int, default=5)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    c32, c16 = caches(32, rng), caches(16, rng)
    ix = {q: jnp.asarray(indices("random", q, rng)) for q in (32, 64, 128, 256)}

    for q in (32, 64, 128, 256):
        point(a, *c32, ix[q], sweep="rows", queries=q, rope_words=32, atoms=16,
              pattern="random")
    for rw, c in ((32, c32), (16, c16)):
        for q in (128, 256):
            point(a, *c, ix[q], sweep="bytes", queries=q, rope_words=rw, atoms=16,
                  pattern="random")
    for atoms in (4, 8, 16, 32, 64):
        point(a, *c32, ix[128], sweep="atoms", queries=128, rope_words=32, atoms=atoms,
              pattern="random")
    for pattern in ("shared", "same"):
        point(a, *c32, jnp.asarray(indices(pattern, 128, rng)), sweep="reuse",
              queries=128, rope_words=32, atoms=16, pattern=pattern)

    # Dense reference: read + write the bytes one 128-query chunk's gather moves.
    for rw in (32, 16):
        words = 128 * TOPK * (128 + rw)
        x = jnp.zeros((words // 128, 128), jnp.uint32)
        f = jax.jit(lambda v: v + jnp.uint32(1))
        us, rr = timed(f, (x,), a.iters, a.repeats)
        print("MICRO " + json.dumps(dict(
            sweep="dense", rows=128 * TOPK, rope_bytes=4 * rw, us=round(us, 1),
            us_range=rr, gbps=round(words * 4 / us / 1e3, 1))), flush=True)
    print("MICRO_DONE", flush=True)


if __name__ == "__main__":
    main()
