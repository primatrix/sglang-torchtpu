"""One whole DSA layer (S1-S6): Source / Ported-S (JAX, torch ops) / Target.

A layer is: store index keys, score, top-k (S1-S4), then write the split FP8 KV
and attend over the picked rows (S5-S6). Each side runs as it deploys:

* Source (page 64, ``--source-root`` = extracted ``m4s_source.tar`` plus the fused
  K / Q kernel files): fused K store, Q tail, scorer with
  ``output_physical_slots=True``, then the backend's ``PhysicalTopKValidation``
  and ``sparse_core_tensor_core_dsa`` (fp8-qk-pv, split layout). Aligned extend
  and decode only: unaligned extend runs the source's XLA key chain, not shipped.
* ported_s_jax: ``bf16_indexer_step(physical=True)`` then
  ``fp8_attention_step(topk_is_physical=True)`` on page-128 tables.
* ported_s (``--stage torch``): the two torch ops in one compiled graph, as the
  backend runs them with SGLANG_TPU_DSA_BF16_INDEXER=1 and
  SGLANG_TPU_DSA_SOURCE_FP8_ATTENTION=1.
* Target: its FP8 indexer (``key_chain_xla`` W8A8 eps 1e-6, no Hadamard,
  ``dsa_indexer_step``, sequence-local rows) and its sparse MLA (FP8 per-tensor,
  per-query kernel: the q-block prefill kernel needs 93 MB VMEM at 64 heads).

Checks: Source vs Ported-S JAX byte for byte (index cache, slots, split KV,
output); torch ops vs Ported-S JAX byte for byte on aligned cases (unaligned
extend differs by ~1 bf16 ulp in the key store, source_chain.md known
differences, so it is reported, not gated). Quality per side: the indexer's
selection shortfall against f64 scores of its own keys, and the attention
output against f64 attention over the same picked rows on unquantized K/V.
Plus the overlap of Target's picks with Ported-S's. Time: device module time
of the whole layer (xprof). ``--stage jax`` then ``--stage torch``.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_fp8_attention_ab as att  # noqa: E402
import bench_indexer_chain as bc  # noqa: E402
from check_physical_topk import to_physical  # noqa: E402

CASES = ("extend_2048", "extend_unaligned", "decode_b64")
FP, K = 128, 2048


def inputs(name):
    """Indexer inputs (bench_indexer_chain) and attention inputs (bench_fp8_attention_ab).

    Both benches build the same requests (q lens, prefixes, page table) for a name.
    """
    ci, ca = bc.case(name), att.case(name)
    for k in ("kv", "pi", "cu_q"):
        assert np.array_equal(ci[k], ca[k] if k != "kv" else ca["kv"]), k
    assert ci["ppr"] == ca["ppr"]
    return ci, ca


def to_logical(c, slots):
    """Flat framework slots -> sequence-local positions; -1 stays -1."""
    inv = np.full(c["cache"].shape[0], -1, np.int64)
    for s in range(c["S"]):
        inv[c["pi"][s * c["ppr"]:(s + 1) * c["ppr"]]] = np.arange(c["ppr"])
    pos = inv[np.maximum(slots, 0) // FP] * FP + np.maximum(slots, 0) % FP
    return np.where(slots >= 0, pos, -1)


def attention_error(ca, out, slots, rows_bf, queries=24):
    """Output vs f64 attention on unquantized K/V over the same picked rows."""
    lat, rope = rows_bf
    out = np.asarray(out, np.float64)
    errs = []
    for t in np.linspace(0, ca["T"] - 1, min(queries, ca["T"])).astype(int):
        sl = slots[t][slots[t] >= 0]
        ql = np.asarray(ca["q_latent"][t], np.float64)
        qr = np.asarray(ca["q_rope"][t], np.float64)
        logits = (ql @ lat[sl].T + qr @ rope[sl].T) * float(ca["sm_scale"])
        p = np.exp(logits - logits.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        ref = p @ lat[sl]
        errs.append(np.abs(out[t] - ref).max() / max(np.abs(ref).max(), 1e-30))
    return {"max_rel": float(np.max(errs)), "median_rel": float(np.median(errs))}


def overlap(a, b, queries=64):
    """Mean |picks(a) & picks(b)| / |picks(b)| over sampled rows."""
    vals = []
    for t in np.linspace(0, a.shape[0] - 1, min(queries, a.shape[0])).astype(int):
        x, y = set(a[t][a[t] >= 0].tolist()), set(b[t][b[t] >= 0].tolist())
        vals.append(len(x & y) / max(len(y), 1))
    return float(np.mean(vals))


def stage_jax(a):
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    for p in (os.path.dirname(a.source_k), os.path.dirname(a.source_q), a.source_root):
        sys.path.insert(0, p)
    from xplane_summary import summarize

    from sgl_jax.srt.kernels.dsa import indexer as src_idx
    from sgl_jax.srt.kernels.dsa.fp8_kv_cache import SplitFP8KVCache as SrcSplit
    from sglang_tpu.attention.pallas import dsa_bf16_indexer as ix
    from sglang_tpu.attention.pallas import dsa_fp8_attention as port
    from sglang_tpu.attention.pallas import dsa_index_topk as dit
    from sglang_tpu.attention.pallas.dsa_bf16_query import expanded_coefficients, query_tail_xla
    from sglang_tpu.attention.pallas.dsa_key_prologue import key_chain_xla
    from sglang_tpu.attention.pallas.util import quantize_fp8_ue8m0

    src_k = __import__(os.path.basename(a.source_k)[:-3]).fused_fp8_k_projection_scatter_pallas
    src_q = __import__(os.path.basename(a.source_q)[:-3]).fused_q_rope_hadamard
    helpers = att.load_source_helpers(a.source_root)
    H = jnp.asarray(bc.hadamard())
    ok = True

    def timed(f, args, donate, tag):
        """Time ``f``; output i + 1 is the new value of donated arg ``donate[i]``."""
        jf = jax.jit(f, donate_argnums=tuple(donate))
        state = {"a": list(args)}

        def call():
            res = jf(*state["a"])
            for i, d in enumerate(donate):
                state["a"][d] = res[i + 1]
            return res

        sync = lambda: jax.block_until_ready(state["a"][donate[0]])
        # To host before the timed calls donate (and delete) the returned caches.
        first = tuple(np.asarray(x) for x in call())
        out = {"us": att._median_time(call, sync, a.iters, a.repeats)[0]}
        if a.xprof_dir:
            d = f"{a.xprof_dir}/layer_{tag}"
            sync()
            jax.profiler.start_trace(d)
            for _ in range(a.iters):
                call()
            sync()
            jax.profiler.stop_trace()
            try:
                x = summarize(d, a.iters)
                out["module_us"] = round(sum(u for _, u in x.get("modules", [])), 1)
                out["top_ops"] = [(n[:60], round(u, 1)) for n, u in x.get("ops", [])[:6]]
            except Exception as e:  # report, do not hide
                out["xprof_error"] = repr(e)[:500]
        return first, out

    for name in a.cases:
        ci, ca = inputs(name)
        prefill, decode = ci["prefill"], ci["decode"]
        P, T, S = ca["P"], ca["T"], ca["S"]
        row = {"case": name, "T": T}
        ji = {k: jnp.asarray(ci[k]) for k in ("hidden", "wk", "ws", "nw", "nb", "query", "cos",
                                              "sin", "w", "kv", "pi", "cu_q", "dist")}
        ja = {k: jnp.asarray(ca[k]) for k in ("q_latent", "q_rope", "kv_c", "k_pe")}
        sm = jnp.float32(ca["sm_scale"])
        rows_bf = att.bf16_rows(ca)

        # Prefilled split FP8 KV, written by the source writer (as bench_fp8_attention_ab).
        n = P * FP
        empty = SrcSplit(jnp.zeros((2 * P, 64, 4, 128), jnp.uint8),
                         jnp.zeros((2 * P, 16, 4, 128), jnp.uint8))
        cache0 = jax.jit(lambda cc, kv_, pe_: helpers["_scatter_fused_kv_paged"](
            cc, kv_, pe_, jnp.array([n], jnp.int32), jnp.arange(2 * P, dtype=jnp.int32),
            jnp.array([0, n], jnp.int32), jnp.array([0, n], jnp.int32), kv_lora_rank=att.LAT,
            fp8_dtype=jnp.float8_e4m3fn))(
            empty, jnp.asarray(ca["prefix_kv"]), jnp.asarray(ca["prefix_pe"]))
        lat0, rope0 = np.asarray(cache0.latent), np.asarray(cache0.rope)

        idx_args = (ji["hidden"], ji["wk"], ji["ws"], ji["nw"], ji["nb"], ji["query"], ji["cos"],
                    ji["sin"], ji["w"], ji["kv"], ji["pi"], ji["cu_q"], ji["dist"])
        att_args = (ja["q_latent"], ja["q_rope"], ja["kv_c"], ja["k_pe"])

        # Ported-S, JAX.
        def ported(icache, lat, rope, hidden, wk, ws, nw, nb, query, cos, sin, w, kv, pi, cu_q,
                   dist, ql, qr, kvc, kpe, sm_):
            slots, icache = ix.bf16_indexer_step(
                icache, hidden, wk, ws, nw, nb, query, cos, sin, w, kv, pi, cu_q, dist, H,
                topk=K, prefill=prefill, decode=decode, physical=True)
            out, lat, rope = port.fp8_attention_step(
                lat, rope, ql, qr, kvc, kpe, slots, kv, pi, cu_q, sm_, allow_fp8_q=True,
                is_prefill=prefill, topk_is_physical=True)
            return out, icache, lat, rope, slots

        p_args = (jnp.asarray(ci["cache"]), jnp.asarray(lat0.reshape(P, FP, 4, 128)),
                  jnp.asarray(rope0.reshape(P, FP // 4, 4, 128)), *idx_args, *att_args, sm)
        (po, pic, plat, prope, pslots), row["ported_s_jax"] = timed(ported, p_args, (0, 1, 2),
                                                                    f"{name}_ported")
        po, pic, plat, prope, pslots = map(np.asarray, (po, pic, plat, prope, pslots))
        q_idx = np.asarray(ix.index_query_tail(ji["query"], ji["cos"], ji["sin"], H,
                                               prefill=prefill), np.float64)
        row["ported_s_jax"]["selection"] = bc.shortfall(
            ci, to_logical(ci, pslots), bc.framework_keys(ci, pic), q_idx)
        row["ported_s_jax"]["attention"] = attention_error(ca, po, pslots, rows_bf)
        if a.npz_dir:
            np.savez(os.path.join(a.npz_dir, f"layer_{name}.npz"), lat0=lat0, rope0=rope0,
                     out=po.view(np.uint16), icache=pic.view(np.int16), lat=plat, rope=prope,
                     slots=pslots)

        # Source, page 64, as deployed.
        table64 = np.stack([2 * ci["pi"], 2 * ci["pi"] + 1], 1).reshape(-1).astype(np.int32)
        cukv = (np.arange(S + 1) * ci["ppr"] * FP).astype(np.int32)
        if decode:
            tp = np.array([table64[s * 2 * ci["ppr"] + (ci["kv"][s] - 1) // 64] * 64
                           + (ci["kv"][s] - 1) % 64 for s in range(S)], np.int32)
        elif T % 64 == 0 and all(p % 64 == 0 for p in ci["prefixes"]):
            tp = np.concatenate([
                table64[s * 2 * ci["ppr"] + (ci["prefixes"][s] + np.arange(0, q, 64)) // 64]
                for s, q in enumerate(np.diff(ci["cu_q"]))]).astype(np.int32)
        else:
            tp = None
        if tp is None:
            row["source"] = {"note": "unaligned extend: the source key store is its XLA chain, "
                                     "not shipped"}
        else:
            tp_j, t64, cukv_j = jnp.asarray(tp), jnp.asarray(table64), jnp.asarray(cukv)

            # Page tables as runtime arguments, as deployed: closed-over jnp arrays
            # become HLO constants, and with them the radix top-k SparseCore lowering
            # failed (RET_CHECK buffer_found, exp-eugdo5kysz).
            def source(icache, lat, rope, hidden, wk, ws, nw, nb, query, cos, sin, w, kv, pi,
                       cu_q, dist, ql, qr, kvc, kpe, sm_, tp_j, t64, cukv_j):
                del pi
                view = src_k(hidden, wk, ws, nw, nb, cos, sin, H, icache, tp_j, page_size=64,
                             token_slots=decode)
                if prefill:
                    ec, es = expanded_coefficients(cos, sin)
                    q = src_q(query, ec, es, H, token_tile=64)
                else:
                    q = query_tail_xla(query, cos, sin, H)
                slots = src_idx.compute_scores_and_select_topk_indices(
                    q, w, view, kv, t64, cu_q, cukv_j, dist, k=K, pages_per_seq=2 * ci["ppr"],
                    one_token_per_seq=decode, topk_impl="radix", output_physical_slots=True)
                out, cc, _ = att.source_run(
                    helpers, SrcSplit(lat, rope), ql, qr, kvc, kpe, slots, kv, t64, cu_q,
                    cukv_j, sm_, allow_fp8_q=True, is_prefill=prefill, topk_is_physical=True)
                return out, view, cc.latent, cc.rope, slots

            s_args = (jnp.asarray(ci["cache"]).reshape(-1, 64, bc.D), jnp.asarray(lat0),
                      jnp.asarray(rope0), *idx_args, *att_args, sm, tp_j, t64, cukv_j)
            try:
                (so, sic, slat, srope, sslots), row["source"] = timed(
                    source, s_args, (0, 1, 2), f"{name}_source")
            except Exception as e:  # report, do not hide; other sides still run
                row["source"] = {"error": repr(e)[:1500]}
                ok = False
                tp = None
        if tp is not None:
            so, sic, slat, srope, sslots = map(np.asarray, (so, sic, slat, srope, sslots))
            row["source"]["attention"] = attention_error(ca, so, sslots, rows_bf)
            ex = {
                "index_cache": np.array_equal(sic.view(np.int16).reshape(-1),
                                              pic.view(np.int16).reshape(-1)),
                "slots": np.array_equal(sslots, pslots),
                "split_kv": np.array_equal(slat.reshape(-1), plat.reshape(-1))
                and np.array_equal(srope.reshape(-1), prope.reshape(-1)),
                "out": np.array_equal(so.view(np.uint16), po.view(np.uint16)),
            }
            row["ported_s_jax_vs_source"] = {k: bool(v) for k, v in ex.items()}
            ok &= all(ex.values())

        # Target, page 128: its FP8 indexer (logical rows) and sparse MLA (per-query).
        try:
            shape = dit.index_key_cache_shape(P, FP, bc.D)
            kf, _, kb = quantize_fp8_ue8m0(jnp.asarray(ci["cache"]).reshape(-1, bc.D))
            ticache0 = dit._store_keys(
                jnp.zeros(shape, jnp.uint8), kf, kb, jnp.array([n], jnp.int32),
                jnp.arange(P, dtype=jnp.int32), jnp.array([0, n], jnp.int32),
                jnp.array([0, 0, 1], jnp.int32))
            fill, t_step = att.target_fns(ca, qblock=False)
            tn, tr = jax.jit(fill, static_argnums=(4,))(
                jnp.zeros((P, FP, 4, 128), jnp.uint8), jnp.zeros((P, FP // 4, 4, 128), jnp.uint8),
                jnp.asarray(ca["prefix_kv"]), jnp.asarray(ca["prefix_pe"]), n,
                jnp.arange(P, dtype=jnp.int32))
            wk_kn = ji["wk"].T
            q_rope_idx = jnp.asarray(np.asarray(
                query_tail_xla(ji["query"], ji["cos"], ji["sin"], jnp.eye(bc.D)),
                np.float32).astype(ci["query"].dtype))

            def target(icache, nope, rope, hidden, query, w, kv, pi, cu_q, dist, ql, qr, kvc,
                       kpe, sm_):
                key = key_chain_xla(hidden, wk_kn, ji["ws"], ji["nw"].astype(jnp.float32),
                                    ji["nb"].astype(jnp.float32), ji["cos"], ji["sin"], eps=1e-6)
                topk, icache = dit.dsa_indexer_step(query, key, w, icache, kv, pi, cu_q, dist,
                                                    weight_scale=1.0, topk=K)
                out, nope, rope = t_step(nope, rope, ql, qr, kvc, kpe, topk, kv, pi, cu_q, dist,
                                         sm_)
                return out, icache, nope, rope, topk

            t_args = (ticache0, tn, tr, ji["hidden"], q_rope_idx, ji["w"], ji["kv"], ji["pi"],
                      ji["cu_q"], ji["dist"], *att_args, sm)
            (to, tic, _, _, trows), row["target"] = timed(target, t_args, (0, 1, 2),
                                                          f"{name}_target")
            to, tic, trows = np.asarray(to), np.asarray(tic), np.asarray(trows)
            rec = tic.reshape(-1, shape[-1])
            deq = (rec[:, :bc.D].view(ci["wk"].dtype).astype(np.float64)
                   * np.exp2(rec[:, bc.D:bc.D + 1].astype(np.float64) - 127)).reshape(-1, FP, bc.D)
            tslots = to_physical(trows, ci["pi"], ci["ppr"], ci["cu_q"])
            row["target"].update(
                kernel="FP8 UE8M0 indexer + per-query sparse MLA (FP8 per-tensor k_scale=1)",
                selection_own_fp8=bc.shortfall(ci, trows, bc.framework_keys(ci, deq),
                                               np.asarray(q_rope_idx, np.float64)),
                attention=attention_error(ca, to, tslots, rows_bf),
                picks_overlap_with_ported_s=overlap(tslots, pslots))
        except Exception as e:  # report, do not hide
            row["target"] = {"error": repr(e)[:3000]}
        print("LAYER " + json.dumps(row), flush=True)
    print("LAYER_JAX_OK" if ok else "LAYER_JAX_FAILED", flush=True)
    return 0 if ok else 1


def stage_torch(a):
    os.environ["JAX_PLATFORMS"] = "cpu"  # TorchTPU owns the device
    import torch
    import torch_tpu  # noqa: F401

    from bench_torch_ab import _profile
    from sglang_tpu.dsa_bf16_indexer_op import build_dsa_bf16_indexer_op
    from sglang_tpu.dsa_fp8_attention_op import build_dsa_fp8_attention_op

    iop, aop = build_dsa_bf16_indexer_op(), build_dsa_fp8_attention_op()
    sync = torch.accelerator.synchronize
    t = lambda x: bc._to_torch(torch, x)
    H = torch.from_numpy(bc.hadamard()).to("tpu")
    ok = True
    for name in a.cases:
        ci, ca = inputs(name)
        ref = np.load(os.path.join(a.npz_dir, f"layer_{name}.npz"))
        P = ca["P"]
        prefill, decode = ci["prefill"], ci["decode"]
        ins = (tuple(t(ci[k]) for k in ("hidden", "wk", "ws", "nw", "nb", "query", "cos", "sin",
                                        "w", "kv", "pi", "cu_q", "dist"))
               + (H,) + tuple(t(ca[k]) for k in ("q_latent", "q_rope", "kv_c", "k_pe")))
        sm = float(ca["sm_scale"])

        def layer(icache, lat, rope, hidden, wk, ws, nw, nb, query, cos, sin, w, kv, pi, cu_q,
                  dist, h, ql, qr, kvc, kpe):
            slots = iop(icache, hidden, wk, ws, nw, nb, query, cos, sin, w, kv, pi, cu_q, dist,
                        h, K, prefill, decode, True)
            return aop(lat, rope, ql, qr, kvc, kpe, slots, kv, pi, cu_q, sm, True, prefill, True)

        fn = torch.compile(layer, backend="tpu", fullgraph=True, dynamic=False)
        fresh = lambda: (t(ci["cache"]), t(ref["lat0"].reshape(P, FP, 4, 128)),
                         t(ref["rope0"].reshape(P, FP // 4, 4, 128)))
        ic, lat, rope = fresh()
        out = fn(ic, lat, rope, *ins).cpu().view(torch.int16).numpy().view(np.uint16)
        sync()
        ex = {
            "out": np.array_equal(out, ref["out"]),
            "index_cache": np.array_equal(ic.cpu().view(torch.int16).numpy(), ref["icache"]),
            "split_kv": np.array_equal(lat.cpu().numpy().reshape(-1), ref["lat"].reshape(-1))
            and np.array_equal(rope.cpu().numpy().reshape(-1), ref["rope"].reshape(-1)),
        }
        row = {"case": name, "torch_vs_ported_s_jax": {k: bool(v) for k, v in ex.items()}}
        if name == "extend_unaligned":
            row["note"] = "known ~1 bf16 ulp key-store difference; reported, not gated"
        else:
            ok &= all(ex.values())
        bargs = (*fresh(), *ins)
        row["us"], _ = att._median_time(lambda: fn(*bargs), sync, a.iters, a.repeats)
        if a.xprof_dir:
            prof = _profile(fn, (*fresh(), *ins), a.iters, f"{a.xprof_dir}/layer_torch_{name}")
            row["module_us"] = round(sum(u for _, u in prof.get("modules", [])), 1)
            row["top_ops"] = [(n[:60], round(u, 1)) for n, u in prof.get("ops", [])[:6]]
        print("LAYERTORCH " + json.dumps(row), flush=True)
    print("LAYER_TORCH_OK" if ok else "LAYER_TORCH_FAILED", flush=True)
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("jax", "torch"), required=True)
    p.add_argument("--cases", nargs="+", default=list(CASES))
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--npz-dir", required=True)
    p.add_argument("--xprof-dir", default="")
    p.add_argument("--source-k", default="/tmp/source/fused_fp8_k_projection_scatter.py")
    p.add_argument("--source-q", default="/tmp/source/fused_q_rope_hadamard.py")
    p.add_argument("--source-root", default="/tmp/source/m4s")
    a = p.parse_args()
    os.makedirs(a.npz_dir, exist_ok=True)
    sys.exit(stage_jax(a) if a.stage == "jax" else stage_torch(a))


if __name__ == "__main__":
    main()
