"""M4-S baseline: Source vs Ported-S split-FP8 KV write + fp8-qk-pv exact attention.

* Source: the pinned source package (``--source-root``, an extracted
  ``m4s_source.tar``). ``dsa_sparse_backend.py`` imports the source model
  framework, which is not shipped, so its three module-level helpers
  (``_scatter_fused_kv_paged``, ``_logical_topk_to_physical_slots``,
  ``_pad_first_axis``) are taken from the file by the line ranges in
  ``source_dsa/MANIFEST.json`` (sha256 checked) and run with the source
  modules' names. The chain is the body of ``DSASparseAttentionBackend._run``
  for the FP8 delivery (split layout, ``DsaConfig("fp8-qk-pv")``) at page 64.
* ported_s_jax: ``dsa_fp8_attention.fp8_attention_step`` on the framework's
  page-128 tables (framework page p = source pages 2p, 2p+1, cache reshaped).

Both sides get the same cache memory and inputs and must store byte-identical
split caches and return byte-identical attention outputs. Accuracy: each side's
output against an f64 reference on the dequantized cache (sampled queries).
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

H, LAT, ROPE, K, FP = 64, 512, 64, 2048, 128

# name -> (q_lens, prefixes, mode); mode: prefill | decode
CASES = {
    "extend_2048": ([2048], [8192], "prefill"),
    "extend_unaligned": ([2000], [8200], "prefill"),
    "decode_b64": ([1] * 64, [8191 + 37 * s for s in range(64)], "decode"),
}


def case(name, seed=0):
    q_lens, prefixes, mode = CASES[name]
    r = np.random.default_rng(seed + sum(q_lens) + len(q_lens))
    S, T = len(q_lens), int(sum(q_lens))
    kv = np.add(prefixes, q_lens).astype(np.int32)
    ppr = int(-(-kv.max() // FP))
    # Request s owns framework pages in reverse order; page 0 is never real.
    pi = np.concatenate([np.arange(1 + (s + 1) * ppr - 1, s * ppr, -1) for s in range(S)])
    cu_q = np.r_[0, np.cumsum(q_lens)].astype(np.int32)
    seq = np.searchsorted(cu_q[1:], np.arange(T), side="right")
    # Sequence-local top-k: each query's own causal window, valid prefix then -1.
    topk = np.full((T, K), -1, np.int32)
    for t in range(T):
        s = seq[t]
        n = kv[s] - q_lens[s] + (t - cu_q[s]) + 1
        m = min(n, K)
        topk[t, :m] = r.choice(n, m, replace=False)
    import ml_dtypes

    bf = ml_dtypes.bfloat16
    P = 1 + S * ppr
    return dict(
        name=name, mode=mode, S=S, T=T, ppr=ppr, kv=kv, pi=pi.astype(np.int32), cu_q=cu_q,
        topk=topk, seq=seq, q_lens=np.asarray(q_lens), prefixes=np.asarray(prefixes), P=P,
        prefix_kv=r.normal(0, 1, (P * FP, LAT)).astype(bf),
        prefix_pe=r.normal(0, 1, (P * FP, ROPE)).astype(bf),
        q_latent=r.normal(0, 1, (T, H, LAT)).astype(bf),
        q_rope=r.normal(0, 1, (T, H, ROPE)).astype(bf),
        kv_c=r.normal(0, 1, (T, LAT)).astype(bf),
        k_pe=r.normal(0, 1, (T, ROPE)).astype(bf),
        # GLM-5.2 qk head dim 192 (nope 128 + rope 64).
        sm_scale=np.float32(192 ** -0.5),
    )


def _median_time(call, sync, iters, repeats):
    call()
    sync()
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            call()
        sync()
        runs.append((time.perf_counter() - t0) / iters * 1e6)
    return statistics.median(runs), runs


def load_source_helpers(source_root):
    """The three source backend helpers, by manifest line range, sha256 checked."""
    from sgl_jax.srt.kernels.dsa import fp8_kv_cache as f8
    from sgl_jax.srt.layers.attention import dsa_cache_ops

    import jax
    import jax.numpy as jnp

    m = json.load(open(os.path.join(
        REPO, "sglang_tpu/attention/pallas/source_dsa/MANIFEST.json")))
    ext = m["extracts"]["layers/attention/dsa_sparse_backend.py"]["functions"]
    lines = open(os.path.join(source_root, "sgl_jax/srt/layers/attention/dsa_sparse_backend.py"),
                 encoding="utf-8").read().splitlines(keepends=True)
    ns = dict(jax=jax, jnp=jnp, SplitFP8KVCache=f8.SplitFP8KVCache,
              quantize_fp8_kv_cache=f8.quantize_fp8_kv_cache,
              quantize_split_fp8_kv_cache=f8.quantize_split_fp8_kv_cache,
              scatter_paged_cache=dsa_cache_ops.scatter_paged_cache)
    for name, info in ext.items():
        a, b = info["lines"]
        text = "".join(lines[a - 1:b])
        # The manifest hashes the extract without its final newline.
        got = hashlib.sha256(text.rstrip("\n").encode()).hexdigest()
        if got != info["sha256"]:
            raise SystemExit(f"source {name} sha256 {got} != manifest {info['sha256']}")
        exec(compile("from __future__ import annotations\n" + text, name, "exec"), ns)
    return ns


def source_run(src, cache, ql, qr, kvc, kpe, topk, seq_lens, pi, cuq, cukv, sm_scale, *,
               allow_fp8_q, is_prefill):
    """``DSASparseAttentionBackend._run`` (FP8 delivery) with the source modules."""
    from sgl_jax.srt.configs.dsa_config import DsaConfig
    from sgl_jax.srt.kernels.dsa.exact_attention import sparse_core_tensor_core_dsa
    from sgl_jax.srt.kernels.dsa.fp8_kv_cache import cache_primary, map_cache_parts
    import jax.numpy as jnp

    page_size = cache_primary(cache).shape[1]
    cache4d = src["_scatter_fused_kv_paged"](
        cache, kvc, kpe, seq_lens, pi, cuq, cukv, kv_lora_rank=LAT, fp8_dtype=jnp.float8_e4m3fn)
    slots, counts = src["_logical_topk_to_physical_slots"](topk, seq_lens, pi, cuq, cukv, page_size)
    plan = DsaConfig(precision="fp8-qk-pv").resolve(
        ql.shape[0], eligible=allow_fp8_q, is_prefill=is_prefill, fp8_kv=True)
    bq_sparse, bq, b_topk = plan.tiles
    pad = src["_pad_first_axis"]
    flat = map_cache_parts(lambda part: part.reshape(-1, *part.shape[2:]), cache4d)
    kw = {} if plan.pipeline == "overlap" else {"split_gather_buffers": plan.pipeline == "split"}
    out = sparse_core_tensor_core_dsa(
        pad(ql, bq_sparse), pad(qr, bq_sparse), flat, pad(slots, bq_sparse),
        pad(counts, bq_sparse), sm_scale, bq_sparse=bq_sparse, bq=bq, b_topk=b_topk,
        fp8_dtype=jnp.float8_e4m3fn, fp8_q=plan.fp8_q, fp8_pv=plan.fp8_pv, **kw)
    return out[: ql.shape[0]], cache4d, str(plan)


def dequant_rows(latent_u8, rope_u8):
    """Framework-page split cache -> f64 latent [N, 512], rope [N, 64] per flat slot."""
    import ml_dtypes

    lat = latent_u8.reshape(-1, 512)
    rope = rope_u8.reshape(-1, 128)
    scale = rope[:, ROPE:ROPE + 4].copy().view(np.float32).astype(np.float64)
    f = lambda x: x.view(ml_dtypes.float8_e4m3fn).astype(np.float64)
    return f(lat) * scale, f(rope[:, :ROPE].copy()) * scale


def reference_error(c, out, latent_u8, rope_u8, queries=24):
    """Max relative error of ``out`` vs f64 attention on the stored (dequantized) cache."""
    lat, rope = dequant_rows(latent_u8, rope_u8)
    out = np.asarray(out, np.float64)
    errs = []
    for t in np.linspace(0, c["T"] - 1, min(queries, c["T"])).astype(int):
        s = c["seq"][t]
        pos = c["topk"][t][c["topk"][t] >= 0]
        tab = c["pi"][s * c["ppr"]:(s + 1) * c["ppr"]]
        slots = tab[pos // FP] * FP + pos % FP
        ql = np.asarray(c["q_latent"][t], np.float64)
        qr = np.asarray(c["q_rope"][t], np.float64)
        logits = (ql @ lat[slots].T + qr @ rope[slots].T) * float(c["sm_scale"])
        p = np.exp(logits - logits.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        ref = p @ lat[slots]
        errs.append(np.abs(out[t] - ref).max() / max(np.abs(ref).max(), 1e-30))
    return {"max_rel": float(np.max(errs)), "median_rel": float(np.median(errs))}


def stage_jax(a):
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp

    sys.path.insert(0, a.source_root)
    from xplane_summary import summarize

    from sgl_jax.srt.kernels.dsa.fp8_kv_cache import SplitFP8KVCache as SrcSplit
    from sglang_tpu.attention.pallas import dsa_fp8_attention as port

    src = load_source_helpers(a.source_root)

    def timed(f, args, tag):
        jf = jax.jit(f)
        out = jf(*args)
        sync = lambda: jax.block_until_ready(jf(*args))
        res = {}
        res["us"], res["runs"] = _median_time(lambda: jf(*args), sync, a.iters, a.repeats)
        if a.xprof_dir:
            d = f"{a.xprof_dir}/fp8att_{tag}"
            sync()
            jax.profiler.start_trace(d)
            for _ in range(a.iters):
                o = jf(*args)
            jax.block_until_ready(o)
            jax.profiler.stop_trace()
            try:
                x = summarize(d, a.iters)
                res["module_us"] = round(sum(u for _, u in x.get("modules", [])), 1)
                res["top_ops"] = [(n[:60], round(u, 1)) for n, u in x.get("ops", [])[:6]]
            except Exception as e:  # report, do not hide
                res["xprof_error"] = repr(e)[:500]
        return out, res

    for name in a.cases:
        c = case(name)
        allow, prefill = True, c["mode"] == "prefill"
        # Initial cache: every slot holds a quantized prefix row (the source writer).
        P = c["P"]
        empty = SrcSplit(jnp.zeros((2 * P, 64, 4, 128), jnp.uint8),
                         jnp.zeros((2 * P, 16, 4, 128), jnp.uint8))
        n = P * FP
        all_pages = jnp.arange(2 * P, dtype=jnp.int32)
        cache0 = jax.jit(lambda cc, kv_, pe_: src["_scatter_fused_kv_paged"](
            cc, kv_, pe_, jnp.array([n], jnp.int32), all_pages, jnp.array([0, n], jnp.int32),
            jnp.array([0, n], jnp.int32), kv_lora_rank=LAT, fp8_dtype=jnp.float8_e4m3fn))(
            empty, jnp.asarray(c["prefix_kv"]), jnp.asarray(c["prefix_pe"]))
        lat0, rope0 = np.asarray(cache0.latent), np.asarray(cache0.rope)

        j = {k: jnp.asarray(c[k]) for k in ("q_latent", "q_rope", "kv_c", "k_pe", "topk", "kv",
                                             "pi", "cu_q")}
        pi64 = port.source_page_table(j["pi"])
        cukv = jnp.arange(c["S"] + 1, dtype=jnp.int32) * (c["ppr"] * FP)
        sm = jnp.float32(c["sm_scale"])
        row = {"case": name, "T": c["T"], "mode": c["mode"]}

        def s_fn(lat, rope, *x):
            out, cc, _ = source_run(src, SrcSplit(lat, rope), *x, allow_fp8_q=allow,
                                    is_prefill=prefill)
            return out, cc.latent, cc.rope

        s_args = (jnp.asarray(lat0), jnp.asarray(rope0), j["q_latent"], j["q_rope"], j["kv_c"],
                  j["k_pe"], j["topk"], j["kv"], pi64, j["cu_q"], cukv, sm)
        try:
            (so, sl, sr), row["source"] = timed(s_fn, s_args, f"{name}_source")
            _, _, row["plan"] = source_run(src, SrcSplit(*s_args[:2]), *s_args[2:],
                                           allow_fp8_q=allow, is_prefill=prefill)
        except Exception as e:  # report, do not hide
            row["source"] = {"error": repr(e)[:3000]}
            print("FP8ATT " + json.dumps(row), flush=True)
            continue

        def p_fn(lat, rope, *x):
            return port.fp8_attention_step(lat, rope, *x, allow_fp8_q=allow, is_prefill=prefill)

        p_args = (jnp.asarray(lat0.reshape(P, FP, 4, 128)),
                  jnp.asarray(rope0.reshape(P, FP // 4, 4, 128)), j["q_latent"], j["q_rope"],
                  j["kv_c"], j["k_pe"], j["topk"], j["kv"], j["pi"], j["cu_q"], sm)
        try:
            (po, pl_, pr), row["ported_s_jax"] = timed(p_fn, p_args, f"{name}_ported")
        except Exception as e:  # report, do not hide
            row["ported_s_jax"] = {"error": repr(e)[:3000]}
            print("FP8ATT " + json.dumps(row), flush=True)
            continue

        so, po = np.asarray(so), np.asarray(po)
        sl, sr = np.asarray(sl), np.asarray(sr)
        pl_, pr = np.asarray(pl_), np.asarray(pr)
        row["cache_exact"] = bool(np.array_equal(sl.reshape(-1), pl_.reshape(-1))
                                  and np.array_equal(sr.reshape(-1), pr.reshape(-1)))
        row["out_exact"] = bool(np.array_equal(so.view(np.uint16), po.view(np.uint16)))
        row["cache_changed"] = not bool(np.array_equal(sl, lat0))
        row["source"]["accuracy"] = reference_error(c, so, sl, sr)
        row["ported_s_jax"]["accuracy"] = reference_error(c, po, pl_, pr)
        for side in ("source", "ported_s_jax"):
            row[side].pop("runs", None)
        print("FP8ATT " + json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=list(CASES))
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--xprof-dir", default="")
    p.add_argument("--source-root", default="/tmp/source/m4s")
    stage_jax(p.parse_args())


if __name__ == "__main__":
    main()
