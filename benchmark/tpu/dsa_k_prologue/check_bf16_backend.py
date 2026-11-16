"""One indexer layer through ``TpuPallasSparseMlaAttnBackend.index_topk_bf16``.

The backend method builds the Hadamard matrix in the graph and passes the
layer's cache and the forward metadata to the chain op. This checks that
plumbing: for each case the backend call (compiled, as the model runs it) must
store the same cache bytes and pick the same top-k as

* the chain torch op called directly with the bench's own arguments, and
* ``dsa_bf16_indexer.bf16_indexer_step`` from JAX (``--stage jax``; aligned
  extend and decode only: unaligned extend has the known ~1 ulp XLA-fallback
  difference recorded in source_chain.md).

Inputs are ``bench_indexer_chain.case`` (page 128, reverse page tables).
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_indexer_chain as bc  # noqa: E402

CASES = ("extend_2048", "extend_unaligned", "decode_b64")


def stage_torch(a):
    os.environ["JAX_PLATFORMS"] = "cpu"
    import torch
    import torch_tpu  # noqa: F401

    from sglang_tpu.attention_pallas_mla_sparse import TpuPallasSparseMlaAttnBackend
    from sglang_tpu.dsa_bf16_indexer_op import build_dsa_bf16_indexer_op

    op = build_dsa_bf16_indexer_op()
    t = lambda x: bc._to_torch(torch, x)
    comp = lambda f: torch.compile(f, backend="tpu", fullgraph=True, dynamic=False)
    H = torch.from_numpy(bc.hadamard()).to("tpu")
    report = {}
    for name in a.cases:
        c = bc.case(name)
        ins = {k: t(c[k]) for k in ("hidden", "wk", "ws", "nw", "nb", "query", "cos", "sin",
                                    "w", "kv", "pi", "cu_q", "dist")}

        # Backend path: a bare backend with only the fields index_topk_bf16 reads.
        cache_b = t(c["cache"])
        backend = TpuPallasSparseMlaAttnBackend.__new__(TpuPallasSparseMlaAttnBackend)
        backend._topk = bc.K
        backend.token_to_kv_pool = SimpleNamespace(get_index_key_cache=lambda layer_id: cache_b)
        backend.forward_metadata = SimpleNamespace(
            kv_lens=ins["kv"], page_indices=ins["pi"], cu_q_lens=ins["cu_q"],
            distribution=ins["dist"])
        batch = SimpleNamespace(spec_info=None)
        params = dict(wk=ins["wk"], wk_scale=ins["ws"], norm_weight=ins["nw"], norm_bias=ins["nb"])

        def via_backend(hidden, query, cos, sin, w):
            return backend.index_topk_bf16(
                forward_batch=batch, layer_id=0, query=query, hidden=hidden, rope_cos=cos,
                rope_sin=sin, head_weights=w, params=params,
                prefill=c["prefill"], decode=c["decode"])

        rows_b = comp(via_backend)(ins["hidden"], ins["query"], ins["cos"], ins["sin"],
                                   ins["w"]).cpu().numpy()

        # Direct op call with the bench's own Hadamard.
        cache_d = t(c["cache"])
        rows_d = comp(lambda cc, *x: op(cc, *x, bc.K, c["prefill"], c["decode"]))(
            cache_d, ins["hidden"], ins["wk"], ins["ws"], ins["nw"], ins["nb"], ins["query"],
            ins["cos"], ins["sin"], ins["w"], ins["kv"], ins["pi"], ins["cu_q"], ins["dist"],
            H).cpu().numpy()

        cb = cache_b.cpu().view(torch.int16).numpy()
        cd = cache_d.cpu().view(torch.int16).numpy()
        np.savez(os.path.join(a.npz_dir, f"backend_{name}.npz"), rows=rows_b, cache=cb)
        report[name] = {
            "backend_vs_op_cache": bool(np.array_equal(cb, cd)),
            "backend_vs_op_topk": bool(np.array_equal(np.sort(rows_b, 1), np.sort(rows_d, 1))),
            "cache_changed": not bool(np.array_equal(cb, c["cache"].view(np.int16))),
        }
        print("BTORCH " + json.dumps({"case": name, **report[name]}), flush=True)


def stage_jax(a):
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    from sglang_tpu.attention.pallas import dsa_bf16_indexer as ix

    H = jnp.asarray(bc.hadamard())
    for name in a.cases:
        c = bc.case(name)
        tor = np.load(os.path.join(a.npz_dir, f"backend_{name}.npz"))
        args = tuple(jnp.asarray(c[k]) for k in ("hidden", "wk", "ws", "nw", "nb", "query",
                                                 "cos", "sin", "w", "kv", "pi", "cu_q", "dist"))
        rows, cache = jax.jit(lambda cc, *x: ix.bf16_indexer_step(
            cc, *x, H, topk=bc.K, prefill=c["prefill"], decode=c["decode"]))(
            jnp.asarray(c["cache"]), *args)
        out = {
            "case": name,
            "backend_vs_jax_cache": bool(np.array_equal(np.asarray(cache).view(np.int16), tor["cache"])),
            "backend_vs_jax_topk": bool(np.array_equal(np.sort(np.asarray(rows), 1),
                                                       np.sort(tor["rows"], 1))),
            "expected_exact": name != "extend_unaligned",
        }
        print("BJAX " + json.dumps(out), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("torch", "jax"), required=True)
    p.add_argument("--cases", nargs="+", default=list(CASES))
    p.add_argument("--npz-dir", default="/tmp/backend")
    a = p.parse_args()
    os.makedirs(a.npz_dir, exist_ok=True)
    stage_torch(a) if a.stage == "torch" else stage_jax(a)


if __name__ == "__main__":
    main()
