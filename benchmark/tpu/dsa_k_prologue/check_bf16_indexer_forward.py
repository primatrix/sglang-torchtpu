"""One indexer layer through the real ``indexer_forward`` (SGLANG_TPU_DSA_BF16_INDEXER=1).

check_bf16_backend.py called ``index_topk_bf16`` directly. This goes one level
up, through the code the model runs: ``indexer_forward`` -> ``bf16_indexer_params``
(``wk`` transposed from the target's ``[hidden, 128]``, LayerNorm params to bf16,
cached on the indexer) -> ``fused_key_rope_tables`` (cos / sin from the rope's
``cos_sin_cache``) -> ``bf16_indexer_modes`` -> ``index_topk_bf16``.

The indexer is the target's layout built from the bench's case: ``wq_b`` and
``weights_proj`` return the case's query and head weights, ``wk`` holds the
case's FP8 weight transposed to ``[hidden, 128]`` with its per-channel scale,
``k_norm`` is sglang's f32 LayerNorm with the case's (bf16-valued) params, and
the rope's ``cos_sin_cache`` row ``t`` is the case's ``cos[t] | sin[t]`` at
position ``t``. The reference is the chain torch op called directly with the
case's arguments and the ``(prefill, decode)`` the source assigns the mode.

Every mode the model sends is run: EXTEND (fused query tail), MIXED,
TARGET_VERIFY and DRAFT_EXTEND_V2 (extend, unfused tail), DECODE. Both sides are
torch ops, so cache bytes and top-k must match exactly in every case. Also
checked: params are computed once per indexer, and a non-GLM-5.2 indexer fails.
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

# (case, forward mode name, source prefill, source decode)
RUNS = (
    ("extend_2048", "EXTEND", True, False),
    ("extend_unaligned", "EXTEND", True, False),
    ("extend_2048", "MIXED", False, False),
    ("extend_unaligned", "TARGET_VERIFY", False, False),
    ("extend_2048", "DRAFT_EXTEND_V2", False, False),
    ("decode_b64", "DECODE", False, True),
)


def stage(a):
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["SGLANG_TPU_DSA_BF16_INDEXER"] = "1"
    import torch
    import torch_tpu  # noqa: F401

    from sglang.srt.layers.layernorm import LayerNorm
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang_tpu import dsa_indexer_tpu as dit
    from sglang_tpu.attention_pallas_mla_sparse import TpuPallasSparseMlaAttnBackend
    from sglang_tpu.dsa_bf16_indexer_op import build_dsa_bf16_indexer_op

    op = build_dsa_bf16_indexer_op()
    t = lambda x: bc._to_torch(torch, x)
    comp = lambda f: torch.compile(f, backend="tpu", fullgraph=True, dynamic=False)
    H = torch.from_numpy(bc.hadamard()).to("tpu")
    ok = True

    def make_indexer(c, ins, **over):
        k_norm = LayerNorm(bc.D, eps=1e-5, dtype=torch.float32).to("tpu")
        with torch.no_grad():
            k_norm.weight.copy_(ins["nw"].float())
            k_norm.bias.copy_(ins["nb"].float())
        wk = SimpleNamespace(weight=ins["wk"].t().contiguous(),  # target layout [hidden, 128]
                             weight_scale=ins["ws"].reshape(-1, 1), bias=None)
        cos_sin = torch.cat([ins["cos"].float(), ins["sin"].float()], -1)  # row t = position t
        query, w = ins["query"], ins["w"]
        ind = SimpleNamespace(
            n_heads=bc.HEADS, head_dim=bc.D, rope_head_dim=64, skip_rope=False,
            is_neox_style=False, use_dsa_indexer_fusion=False, k_norm=k_norm, wk=wk,
            wq_b=lambda q_lora: (query.reshape(q_lora.shape[0], -1), None),
            weights_proj=lambda x: (w, None),
            rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin),
        )
        for k, v in over.items():
            setattr(ind, k, v)
        return ind

    for name, mode_name, pre, dec in RUNS:
        c = bc.case(name)
        ins = {k: t(c[k]) for k in ("hidden", "wk", "ws", "nw", "nb", "query", "cos", "sin",
                                    "w", "kv", "pi", "cu_q", "dist")}
        mode = getattr(ForwardMode, mode_name)
        mapped = dit.bf16_indexer_modes(mode)
        cache_f = t(c["cache"])
        backend = TpuPallasSparseMlaAttnBackend.__new__(TpuPallasSparseMlaAttnBackend)
        backend._kpool = 1
        backend._topk = bc.K
        backend.token_to_kv_pool = SimpleNamespace(get_index_key_cache=lambda layer_id: cache_f)
        backend.forward_metadata = SimpleNamespace(
            kv_lens=ins["kv"], page_indices=ins["pi"], cu_q_lens=ins["cu_q"],
            distribution=ins["dist"])
        batch = SimpleNamespace(forward_mode=mode, spec_info=None)
        indexer = make_indexer(c, ins)
        positions = torch.arange(c["T"], dtype=torch.int64, device="tpu")
        q_lora = torch.zeros((c["T"], 8), dtype=torch.bfloat16, device="tpu")

        def via_forward(x, ql, pos):
            return backend.forward_indexer(indexer, x=x, q_lora=ql, positions=pos,
                                           forward_batch=batch, layer_id=0)

        rows_f = comp(via_forward)(ins["hidden"], q_lora, positions).cpu().numpy()
        params_once = getattr(indexer, dit._BF16_PARAMS_ATTR, None) is not None

        cache_d = t(c["cache"])
        rows_d = comp(lambda cc, *x: op(cc, *x, bc.K, pre, dec))(
            cache_d, ins["hidden"], ins["wk"], ins["ws"], ins["nw"], ins["nb"], ins["query"],
            ins["cos"], ins["sin"], ins["w"], ins["kv"], ins["pi"], ins["cu_q"], ins["dist"],
            H).cpu().numpy()

        cf = cache_f.cpu().view(torch.int16).numpy()
        cd = cache_d.cpu().view(torch.int16).numpy()
        row = {
            "case": name, "mode": mode_name, "mapped_prefill_decode": list(mapped),
            "mapping_ok": mapped == (pre, dec),
            "forward_vs_op_cache": bool(np.array_equal(cf, cd)),
            "forward_vs_op_topk": bool(np.array_equal(np.sort(rows_f, 1), np.sort(rows_d, 1))),
            "cache_changed": not bool(np.array_equal(cf, c["cache"].view(np.int16))),
            "params_cached": params_once,
        }
        ok &= all(row[k] for k in ("mapping_ok", "forward_vs_op_cache", "forward_vs_op_topk",
                                   "cache_changed", "params_cached"))
        print("FWD " + json.dumps(row), flush=True)

    # A layout the source chain does not cover must fail, not fall back to FP8.
    c = bc.case("decode_b64")
    ins = {k: t(c[k]) for k in ("wk", "ws", "nw", "nb", "query", "cos", "sin", "w")}
    for bad in ({"is_neox_style": True}, {"head_dim": 64}, {"skip_rope": True}):
        try:
            dit.bf16_indexer_params(make_indexer(c, ins, **bad))
            res = "no error"
        except ValueError as e:
            res = "ValueError: " + str(e)[:160]
        ok &= res.startswith("ValueError")
        print("REJECT " + json.dumps({"override": bad, "result": res}), flush=True)
    print("FWD_ALL_OK" if ok else "FWD_FAILED", flush=True)
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser()
    p.parse_args()
    sys.exit(stage(None))


if __name__ == "__main__":
    main()
