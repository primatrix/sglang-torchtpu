"""Target vs Ported: one GLM-5.2 indexer step at W-P prefill shapes, through torch_tpu.

* ``target``: the torch chain ``indexer_forward`` runs today: ``apply_tpu_fp8_linear``
  for ``wk``, f32 LayerNorm, the TPU rope path (table gather, +-1 rotation dot),
  then the ``dsa_index_topk`` op (quantize, XLA key store, top-k).
* ``ported``: the rope-table gather of ``fused_key_rope_tables``, then the
  ``dsa_index_topk_fused_key`` op (fused key prologue on aligned pages, top-k).

The query (and its rope) and the head weights are inputs to both: the two sides
compute them identically. Both are ``torch.compile``d (backend "tpu",
fullgraph). Each repeat times ``--iters`` back-to-back calls with a device sync
at the end; the reported value is the median per-call time over ``--repeats``.
This is wall time of a device-bound loop, not an xprof trace.

The batch is ``tokens / CHUNK`` requests of CHUNK new tokens each, extending a
page-aligned prefix, so the fused kernel (not its XLA fallback) runs.
"""

import argparse
import json
import os
import statistics
import time

# TorchTPU owns the device; JAX exports the custom call on the CPU.
os.environ["JAX_PLATFORMS"] = "cpu"

import torch
import torch.nn.functional as F
import torch_tpu  # noqa: F401

from sglang_tpu.dsa_indexer_tpu import (
    build_dsa_index_topk_fused_key_op,
    build_dsa_index_topk_op,
)
from sglang_tpu.fp8_linear import apply_tpu_fp8_linear

HIDDEN, D, ROPE, PAGE, EPS = 6144, 128, 64, 128, 1e-6
DEV = "tpu"


def _metadata(tokens, prefix, chunk):
    seqs = max(1, tokens // chunk)
    q = tokens // seqs
    pages_per_seq = (prefix + q + PAGE - 1) // PAGE
    kv_lens = torch.full((seqs,), prefix + q, dtype=torch.int32)
    page_indices = 1 + torch.arange(seqs * pages_per_seq, dtype=torch.int32)
    cu_q_lens = torch.arange(seqs + 1, dtype=torch.int32) * q
    distribution = torch.tensor([0, 0, seqs], dtype=torch.int32)
    positions = (prefix + torch.arange(q)).repeat(seqs)
    num_pages = seqs * pages_per_seq + 1
    return kv_lens, page_indices, cu_q_lens, distribution, positions, num_pages, seqs


def _time(fn, args, iters, repeats):
    fn(*args)
    torch.accelerator.synchronize()
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(*args)
        torch.accelerator.synchronize()
        runs.append((time.perf_counter() - t0) / iters * 1e6)
    return statistics.median(runs), runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[2048, 8192, 32768])
    p.add_argument("--prefix", type=int, default=49152)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", default="/tmp/torch_ab.json")
    a = p.parse_args()

    gen = torch.Generator().manual_seed(0)
    wk = (torch.randn(HIDDEN, D, generator=gen) * 0.02).to(torch.float8_e4m3fn).to(DEV)
    wk_scale = (torch.rand(D, generator=gen) + 0.5).to(DEV)
    nw = (torch.rand(D, generator=gen) + 0.5).to(DEV)
    nb = (torch.randn(D, generator=gen) * 0.1).to(DEV)
    max_pos = a.prefix + a.chunk
    angle = torch.rand(max_pos, ROPE // 2, generator=gen) * 6.28
    # The rope's cos_sin_cache (bf16 on TPU) and the TPU path's expanded table.
    cos_sin = torch.cat([angle.cos(), angle.sin()], -1).to(torch.bfloat16)
    cos, sin = cos_sin.chunk(2, -1)
    tpu_table = torch.cat(
        [cos.repeat_interleave(2, -1), sin.repeat_interleave(2, -1)], -1
    ).to(DEV)
    rot = torch.zeros(ROPE, ROPE, dtype=torch.bfloat16)
    idx = torch.arange(ROPE)
    rot[idx[1::2], idx[0::2]] = -1.0
    rot[idx[0::2], idx[1::2]] = 1.0
    rot = rot.to(DEV)
    cos_sin = cos_sin.to(DEV)

    topk_op = build_dsa_index_topk_op()
    fused_op = build_dsa_index_topk_fused_key_op()

    def target(query, hidden, positions, head_weights, cache, kv_lens, pages, cu, dist):
        k = apply_tpu_fp8_linear(hidden, wk, wk_scale)
        k = F.layer_norm(k.float(), (D,), nw, nb, EPS).to(torch.bfloat16)
        cs = tpu_table[positions]
        c, s = cs[:, :ROPE], cs[:, ROPE:]
        kr = k[:, :ROPE]
        kr = kr * c + (kr @ rot) * s
        k = torch.cat([kr.to(k.dtype), k[:, ROPE:]], -1)
        return topk_op(
            query, k.contiguous(), head_weights, cache, kv_lens, pages, cu, dist,
            1.0, a.topk, 1, "all", "xla",
        )

    def ported(query, hidden, positions, head_weights, cache, kv_lens, pages, cu, dist):
        c, s = cos_sin.index_select(0, positions).chunk(2, -1)
        return fused_op(
            query, hidden, wk, wk_scale, nw, nb, c.contiguous(), s.contiguous(),
            head_weights, cache, kv_lens, pages, cu, dist,
            torch.zeros(1, dtype=torch.int32, device=DEV),
            EPS, 1.0, a.topk, 1, "all", 0,
        )

    compile_ = lambda f: torch.compile(f, backend="tpu", fullgraph=True, dynamic=False)
    variants = {"target": compile_(target), "ported": compile_(ported)}
    report = {"args": vars(a), "rows": []}
    for m in a.tokens:
        kv_lens, pages, cu, dist, positions, num_pages, seqs = _metadata(m, a.prefix, a.chunk)
        hidden = torch.randn(m, HIDDEN, generator=gen).to(torch.bfloat16).to(DEV)
        query = torch.randn(m, a.heads, D, generator=gen).to(torch.bfloat16).to(DEV)
        head_weights = torch.rand(m, a.heads, generator=gen).to(DEV)
        meta = [t.to(DEV) for t in (kv_lens, pages, cu, dist)]
        positions = positions.to(DEV)
        row = {"tokens": m, "requests": seqs}
        picks = {}
        for name, fn in variants.items():
            # Same random prefix keys for both, so they select over the same cache.
            cache = torch.randint(
                0, 256, (num_pages, PAGE // 4, 4, 256), generator=torch.Generator().manual_seed(m),
                dtype=torch.uint8,
            ).to(DEV)
            args = (query, hidden, positions, head_weights, cache, *meta)
            picks[name] = fn(*args).cpu()
            row[f"{name}_us"], row[f"{name}_runs"] = _time(fn, args, a.iters, a.repeats)
        row["speedup"] = row["target_us"] / row["ported_us"]
        # Sanity, not a criterion: share of each row's picks both sides agree on.
        t, q = picks["target"].sort(1).values, picks["ported"].sort(1).values
        row["pick_overlap_mean"] = float(
            sum(len(set(t[i].tolist()) & set(q[i].tolist())) for i in range(0, m, max(1, m // 256)))
            / (len(range(0, m, max(1, m // 256))) * a.topk)
        )
        print("ROW " + json.dumps(row), flush=True)
        report["rows"].append(row)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
