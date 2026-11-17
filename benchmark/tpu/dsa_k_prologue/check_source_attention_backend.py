"""The backend path of SGLANG_TPU_DSA_SOURCE_FP8_ATTENTION, one attention layer, on TPU.

check_physical_topk.py called the ``dsa_fp8_attention`` op directly. This goes
one level up, through the code the model runs: ``_forward_mla`` ->
``_forward_source_fp8_attention`` (absorbed q / k views made dense, the pool's
split nope / rope buffers, the forward mode mapped to ``(allow_fp8_q,
is_prefill)``, ``layer.scaling``) -> the op with ``topk_is_physical=True``.

The inputs are bench_fp8_attention_ab.py's cases, with the top-k mapped to
flat cache slots as the bf16 indexer emits them (``physical=True``). The output
and the pool's buffers must equal the Ported-S JAX run on the logical top-k
(``--npz-dir`` from that bench) byte for byte: EXTEND for the two prefill cases,
DECODE for decode. A MIXED batch must raise, not attend.
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_fp8_attention_ab as att  # noqa: E402
from check_physical_topk import to_physical  # noqa: E402

RUNS = (("extend_2048", "EXTEND"), ("extend_unaligned", "EXTEND"), ("decode_b64", "DECODE"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz-dir", required=True)
    a = p.parse_args()

    os.environ["JAX_PLATFORMS"] = "cpu"  # TorchTPU owns the device
    import torch
    import torch_tpu  # noqa: F401

    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang_tpu.dsa_fp8_attention_op import build_dsa_fp8_attention_op
    from sglang_tpu.attention_pallas_mla_sparse import TpuPallasSparseMlaAttnBackend

    bf = lambda x: torch.from_numpy(np.ascontiguousarray(x).view(np.int16)).view(torch.bfloat16).to("tpu")
    dev = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to("tpu")
    comp = lambda f: torch.compile(f, backend="tpu", fullgraph=True, dynamic=False)
    ok = True

    for name, mode_name in RUNS:
        c = att.case(name)
        ref = np.load(os.path.join(a.npz_dir, f"fp8att_{name}.npz"))
        P, T = c["P"], c["T"]
        nope = dev(ref["lat0"].reshape(P, att.FP, 4, 128))
        rope = dev(ref["rope0"].reshape(P, att.FP // 4, 4, 128))

        # __new__ skips __init__, which registers the op before compiling.
        build_dsa_fp8_attention_op()
        backend = TpuPallasSparseMlaAttnBackend.__new__(TpuPallasSparseMlaAttnBackend)
        backend._source_fp8_attention = True
        backend._zero_rope = False
        backend.token_to_kv_pool = SimpleNamespace(
            get_nope_cache=lambda layer_id: nope, get_rope_cache=lambda layer_id: rope)
        backend.forward_metadata = SimpleNamespace(
            kv_lens=dev(c["kv"]), page_indices=dev(c["pi"]), cu_q_lens=dev(c["cu_q"]))
        layer = SimpleNamespace(layer_id=0, tp_q_head_num=att.H, v_head_dim=att.LAT,
                                qk_head_dim=att.LAT + att.ROPE, scaling=float(c["sm_scale"]))
        batch = SimpleNamespace(forward_mode=getattr(ForwardMode, mode_name), spec_info=None)
        slots = dev(to_physical(c["topk"], c["pi"], c["ppr"], c["cu_q"]))

        def fwd(q, k, q_rope, k_rope, topk):
            return backend._forward_mla(q, k, layer, batch, q_rope=q_rope, k_rope=k_rope,
                                        topk_indices=topk)

        # The model hands k / k_rope as [T, 1, D] slices; the backend makes them dense.
        out = comp(fwd)(bf(c["q_latent"]), bf(c["kv_c"]).reshape(T, 1, att.LAT),
                        bf(c["q_rope"]), bf(c["k_pe"]).reshape(T, 1, att.ROPE), slots)
        got = out.cpu().view(torch.int16).numpy().view(np.uint16).reshape(T, att.H, att.LAT)
        row = {
            "case": name, "mode": mode_name,
            "out_shape": list(out.shape),
            "out_exact_vs_jax": bool(np.array_equal(got, ref["out"])),
            "cache_exact_vs_jax": bool(
                np.array_equal(nope.cpu().numpy().reshape(-1), ref["latent"].reshape(-1))
                and np.array_equal(rope.cpu().numpy().reshape(-1), ref["rope"].reshape(-1))),
        }
        ok &= row["out_exact_vs_jax"] and row["cache_exact_vs_jax"]
        ok &= row["out_shape"] == [T, att.H * att.LAT]
        print("BACKEND " + json.dumps(row), flush=True)

    # MIXED: the source runs it on the unmigrated HLO-overlap pipeline.
    batch = SimpleNamespace(forward_mode=ForwardMode.MIXED, spec_info=None)
    try:
        backend._forward_mla(bf(c["q_latent"]), bf(c["kv_c"]).reshape(T, 1, att.LAT), layer,
                             batch, q_rope=bf(c["q_rope"]),
                             k_rope=bf(c["k_pe"]).reshape(T, 1, att.ROPE), topk_indices=slots)
        res = "no error"
    except NotImplementedError as e:
        res = "NotImplementedError: " + str(e)[:120]
    ok &= res.startswith("NotImplementedError")
    print("BACKEND_MIXED " + json.dumps({"result": res}), flush=True)
    print("BACKEND_ALL_OK" if ok else "BACKEND_FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
