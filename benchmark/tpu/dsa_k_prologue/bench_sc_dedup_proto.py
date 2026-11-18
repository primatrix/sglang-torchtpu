"""Prototype: gather each 128-query chunk's distinct rows once (extend, PR 2198 shapes).

GLM-5.2's real extend top-k (exp-fj8ahrq5pz, layers 0-60, ~53K context) has only
~3-5% distinct rows per 128-query chunk (128 x 2048 = 262,144 entries ->
~8-16K rows), with 75% overlap between adjacent queries. The SparseCore gather
fit is ~36 us per call + 0.55 ns per row, so gathering only the distinct rows
would cut the per-chunk gather from ~180 us to ~45 us, if the rest is cheap.

The indices here are synthetic with that structure: a pool of ``D`` rows of the
cache; query 0 takes 2048 of them, each next query keeps 75% of the previous
query's rows and draws the rest from the pool. Per chunk, timed separately:

* ``gather_full``: today's gather, every entry (262,144 rows).
* ``unique``: ``jnp.unique(size=D_pad, return_inverse=True)`` on the chunk's
  entries (TensorCore), plus the [128, D_pad] selection mask from the inverse.
* ``gather_distinct``: the native SparseCore gather of the distinct rows only
  (padded to a whole owner round, tail skipped with ``num_live_indices``).
* ``expand_take``: rebuilding the per-query rows from the distinct buffer with
  an XLA take (what an unchanged attention kernel would need).
* ``dense_masked_attn``: attention of the chunk's 128 x 64 heads over all
  ``D_pad`` distinct rows with the selection mask (no per-query rows at all).

Prints one ``DEDUP`` JSON line per (D, part).
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

Q, TOPK, HEADS, CACHE_ROWS = 128, 2048, 64, 51264
KEEP = 0.75


def chunk_indices(pool_size, rng):
    pool = rng.choice(CACHE_ROWS, pool_size, replace=False)
    rows = np.empty((Q, TOPK), np.int32)
    cur = rng.choice(pool, TOPK, replace=False)
    rows[0] = cur
    keep = int(TOPK * KEEP)
    for i in range(1, Q):
        kept = rng.choice(cur, keep, replace=False)
        rest = np.setdiff1d(pool, kept, assume_unique=True)
        cur = np.concatenate([kept, rng.choice(rest, TOPK - keep, replace=False)])
        rows[i] = cur
    return rows


def timed(fn, args, iters, repeats):
    jax.block_until_ready(fn(*args))
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn(*args)
        jax.block_until_ready(out)
        runs.append((time.perf_counter() - t0) / iters * 1e6)
    return round(statistics.median(runs), 1), [round(min(runs), 1), round(max(runs), 1)]


def emit(**row):
    print("DEDUP " + json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pools", type=int, nargs="+", default=[8192, 12288, 16384])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--repeats", type=int, default=5)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    nope = jnp.asarray(rng.integers(0, 2**32, (CACHE_ROWS, 128), dtype=np.uint32))
    rope = jnp.asarray(rng.integers(0, 2**32, (CACHE_ROWS // 4, 128), dtype=np.uint32))
    owner_round = 32 * 64  # owners x occurrences per atom: the gather's size step

    for pool in a.pools:
        rows = chunk_indices(pool, rng)
        distinct = int(np.unique(rows).size)
        d_pad = -(-distinct // owner_round) * owner_round
        base = dict(pool=pool, distinct=distinct, d_pad=d_pad,
                    distinct_frac=round(distinct / rows.size, 4))
        ix = jnp.asarray(rows)

        def run(part, fn, args, **extra):
            try:
                us, rr = timed(fn, args, a.iters, a.repeats)
                emit(**base, part=part, us=us, us_range=rr, **extra)
            except Exception as e:  # report, do not hide
                emit(**base, part=part, error=repr(e)[:600])

        full = jax.jit(lambda n, r, i: nsc.dsa_gather_native_sc(
            n, r, i, out_size=Q * TOPK, atoms_per_batch=16))
        run("gather_full", full, (nope, rope, ix), rows=Q * TOPK)

        def uniq(i):
            u, inv = jnp.unique(i.reshape(-1), size=d_pad, fill_value=-1,
                                return_inverse=True)
            inv = inv.reshape(Q, TOPK)
            mask = jnp.zeros((Q, d_pad), jnp.bool_).at[
                jnp.arange(Q)[:, None], inv].set(True)
            return u, inv, mask

        run("unique", jax.jit(uniq), (ix,))

        uniq_np = np.unique(rows)
        # Tail: distinct dummy rows (a repeated row serializes the gather), and
        # num_live_indices skips them anyway.
        filler = np.setdiff1d(np.arange(CACHE_ROWS), uniq_np)[: d_pad - distinct]
        dix = jnp.asarray(np.concatenate([uniq_np, filler]).astype(np.int32).reshape(-1, 128))
        live = jnp.int32(distinct)
        dist = jax.jit(lambda n, r, i, l: nsc.dsa_gather_native_sc(
            n, r, i, out_size=d_pad, atoms_per_batch=16, num_live_indices=l))
        run("gather_distinct", dist, (nope, rope, dix, live), rows=distinct)

        gn, gr = dist(nope, rope, dix, live)
        inv = jnp.asarray(np.searchsorted(uniq_np, rows).astype(np.int32))
        expand = jax.jit(lambda g, i: jnp.take(g, i.reshape(-1), axis=0))
        run("expand_take", expand, (gn, inv), rows=Q * TOPK)

        q = jnp.asarray(rng.normal(0, 1, (Q, HEADS, 576)), jnp.bfloat16)
        k = jnp.asarray(rng.normal(0, 1, (d_pad, 576)), jnp.bfloat16)
        mask = jnp.asarray(np.zeros((Q, d_pad), bool)).at[
            jnp.arange(Q)[:, None], inv].set(True)

        def dense(q_, k_, m):
            s = jnp.einsum("qhd,kd->qhk", q_, k_, preferred_element_type=jnp.float32)
            s = jnp.where(m[:, None, :], s, -jnp.inf)
            pr = jax.nn.softmax(s, axis=-1).astype(jnp.bfloat16)
            return jnp.einsum("qhk,kd->qhd", pr, k_[:, :512])

        run("dense_masked_attn", jax.jit(dense), (q, k, mask),
            flop=2 * Q * HEADS * d_pad * (576 + 512))
    print("DEDUP_DONE", flush=True)


if __name__ == "__main__":
    main()
