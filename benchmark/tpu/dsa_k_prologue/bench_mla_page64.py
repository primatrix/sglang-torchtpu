"""Target sparse MLA prefill (``sparse_mla_prefill_fused_tokens``) at page 64 vs page 128.

The source chain needs page 64; until S5/S6 migrate, GLM-5.2 keeps this target
kernel, whose query-blocked path requires ``page_size % 128 == 0`` and at most
``QBLOCK_TOKENS_MAX_PAGES_PER_SEQ`` pages per request. This measures what page 64
costs it. Raw JAX, 8 query heads per rank (as test_mla_sparse_fused), top-k
2048 logical positions per token drawn from the prefix, valid FP8 cache bytes.
Speed only: the per-shape kernel path taken is reported, correctness is in the
unit tests.
"""

import argparse
import json
import os
import statistics
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sglang_tpu.attention.pallas import mla_sparse_fused as msf  # noqa: E402

HEADS, NOPE, ROPE, TOPK, CHUNK = 8, 512, 64, 2048, 2048
FP8 = ml_dtypes.float8_e4m3fn


def _fp8_bytes(rng, n):
    return rng.normal(0, 1, n).astype(FP8).view(np.uint8)


def _case(tokens, prefix, page, seed=0):
    rng = np.random.default_rng(seed)
    seqs = max(1, tokens // CHUNK)
    q = tokens // seqs
    pps = -(-(prefix + q) // page)
    num_pages = seqs * pps + 1
    nope = _fp8_bytes(rng, num_pages * page * 512).reshape(num_pages, page, 4, 128)
    rope = _fp8_bytes(rng, num_pages * page * 128).reshape(num_pages, page // 4, 4, 128)
    topk = rng.integers(0, prefix, (tokens, TOPK), dtype=np.int32)
    bf = lambda shape: jnp.asarray(rng.normal(0, 1, shape).astype(ml_dtypes.bfloat16))
    args = dict(
        ql_nope=bf((HEADS, tokens, NOPE)),
        q_pe=bf((HEADS, tokens, ROPE)),
        new_kv_c=jnp.asarray(rng.normal(0, 1, (tokens, NOPE)).astype(FP8)),
        new_k_pe=jnp.asarray(rng.normal(0, 1, (tokens, ROPE)).astype(FP8)),
        kv_lens=jnp.full((seqs,), prefix + q, jnp.int32),
        page_indices=jnp.arange(1, seqs * pps + 1, dtype=jnp.int32),
        cu_q_lens=jnp.arange(seqs + 1, dtype=jnp.int32) * q,
        distribution=jnp.array([0, 0, seqs], jnp.int32),
        topk_indices=jnp.asarray(topk),
    )
    path = (
        "qblock_tokens"
        if tokens >= msf.QBLOCK_MIN_TOKENS
        and tokens % 128 == 0
        and pps <= msf.QBLOCK_TOKENS_MAX_PAGES_PER_SEQ
        and page % 128 == 0
        and page & (page - 1) == 0
        else "per_query"
    )
    return args, jnp.asarray(nope), jnp.asarray(rope), pps, path


def _bench(tokens, prefix, page, iters, repeats, xprof_dir):
    args, nope, rope, pps, path = _case(tokens, prefix, page)

    def step(nope, rope):
        out, nope, rope = msf.sparse_mla_prefill_fused_tokens(
            cache_nope=nope, cache_rope=rope, sm_scale=576**-0.5, k_scale=1.0, **args
        )
        return out, nope, rope

    f = jax.jit(step, donate_argnums=(0, 1))
    state = {"o": f(nope, rope)}
    call = lambda: state.__setitem__("o", f(state["o"][1], state["o"][2]))
    sync = lambda: jax.block_until_ready(state["o"])
    sync()
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            call()
        sync()
        runs.append((time.perf_counter() - t0) / iters * 1e6)
    row = {"tokens": tokens, "prefix": prefix, "page": page, "pages_per_seq": pps,
           "path": path, "us": statistics.median(runs), "runs": runs}
    if xprof_dir:
        from xplane_summary import summarize

        d = f"{xprof_dir}/mla_{tokens}_{prefix}_{page}"
        jax.profiler.start_trace(d)
        for _ in range(iters):
            call()
        sync()
        jax.profiler.stop_trace()
        try:
            x = summarize(d, iters)
            row["module_us"] = sum(u for _, u in x["modules"])
            row["top_ops"] = [(n[:60], round(u, 1)) for n, u in x["ops"][:5]]
        except Exception as e:  # report, do not hide
            row["xprof_error"] = repr(e)[:500]
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[2048, 8192])
    # 8192: page 128 within the q-block page limit; 49152: the W-P prefix, over it.
    p.add_argument("--prefixes", type=int, nargs="+", default=[8192, 49152])
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--xprof-dir", default="")
    p.add_argument("--out", default="/tmp/mla_page64.json")
    a = p.parse_args()
    rows = []
    for prefix in a.prefixes:
        for m in a.tokens:
            for page in (128, 64):
                try:
                    row = _bench(m, prefix, page, a.iters, a.repeats, a.xprof_dir)
                except Exception as e:  # report, do not hide
                    row = {"tokens": m, "prefix": prefix, "page": page, "error": repr(e)[:1500]}
                print("MLA " + json.dumps(row), flush=True)
                rows.append(row)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
