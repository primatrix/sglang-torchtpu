"""Pin the K-prologue numerics to the torch_tpu target chain.

Two processes, because torch_tpu and JAX cannot both own the TPU:

* ``--stage torch``: run the target chain of ``indexer_forward`` on torch_tpu
  (``apply_tpu_fp8_linear``, f32 LayerNorm, interleaved rope), eager and under
  ``torch.compile(backend="tpu")``, and save every intermediate.
* ``--stage jax``: on the same inputs, run ``key_prologue_pallas`` and
  ``quantize_fp8_ue8m0`` of the torch key, and report where they differ.

The report decides which rounding the kernel must follow; it is not a test.
"""

import argparse
import json
import os

import numpy as np

T, HIDDEN, D, ROPE, EPS = 640, 6144, 128, 64, 1e-6


def _inputs():
    import ml_dtypes

    rng = np.random.default_rng(0)
    return dict(
        x=rng.normal(0, 1, (T, HIDDEN)).astype(ml_dtypes.bfloat16),
        wk=rng.normal(0, 0.02, (HIDDEN, D)).astype(ml_dtypes.float8_e4m3fn),
        wk_scale=rng.uniform(0.5, 1.5, D).astype(np.float32),
        nw=rng.uniform(0.5, 1.5, D).astype(np.float32),
        nb=rng.normal(0, 0.1, D).astype(np.float32),
        angle=rng.uniform(0, 2 * np.pi, (T, ROPE // 2)).astype(np.float32),
    )


def stage_torch(out):
    import torch
    import torch.nn.functional as F
    import torch_tpu  # noqa: F401

    from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb
    from sglang_tpu.fp8_linear import _quantize_activation_per_row, apply_tpu_fp8_linear

    inp = _inputs()
    dev = "tpu"
    x = torch.from_numpy(inp["x"].view(np.int16)).view(torch.bfloat16).to(dev)
    wk = torch.from_numpy(inp["wk"].view(np.uint8)).view(torch.float8_e4m3fn).to(dev)
    ws = torch.from_numpy(inp["wk_scale"]).to(dev)
    nw = torch.from_numpy(inp["nw"]).to(dev)
    nb = torch.from_numpy(inp["nb"]).to(dev)
    ang = torch.from_numpy(inp["angle"])
    cos = ang.cos().to(torch.bfloat16).to(dev)
    sin = ang.sin().to(torch.bfloat16).to(dev)

    def chain(x):
        _, x_scale = _quantize_activation_per_row(x, torch.float8_e4m3fn)
        k = apply_tpu_fp8_linear(x, wk, ws)
        k_proj = k
        k = F.layer_norm(k.float(), (D,), weight=nw, bias=nb, eps=EPS).to(torch.bfloat16)
        k_norm = k
        rot = apply_rotary_emb(k[:, None, :ROPE], cos, sin, is_neox_style=False)[:, 0]
        k = torch.cat([rot.to(k.dtype), k[:, ROPE:]], dim=-1)
        return x_scale, k_proj, k_norm, k

    res = {}
    for name, fn in (
        ("eager", chain),
        ("compiled", torch.compile(chain, backend="tpu", fullgraph=True, dynamic=False)),
    ):
        outs = [t.float().cpu().numpy() for t in fn(x)]
        torch.accelerator.synchronize()
        for k, v in zip(("x_scale", "k_proj", "k_norm", "k"), outs):
            res[f"{name}_{k}"] = v
    np.savez(out, **res)
    print("torch stage saved", out, {k: v.shape for k, v in res.items()})


def stage_jax(torch_npz, out_json):
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    import jax
    import jax.numpy as jnp
    import ml_dtypes

    from sglang_tpu.attention.pallas.dsa_key_prologue import key_prologue_pallas
    from sglang_tpu.attention.pallas.util import quantize_fp8_ue8m0

    inp = _inputs()
    tr = dict(np.load(torch_npz))
    x = inp["x"].astype(np.float32)
    amax = np.abs(x).max(1, keepdims=True)
    report = {"jax": jax.__version__, "device": str(jax.devices()[0])}

    for mode in ("eager", "compiled"):
        ts = tr[f"{mode}_x_scale"]
        report[f"{mode}_x_scale_eq_exact_div"] = bool(
            np.array_equal(ts, np.maximum(amax / np.float32(448.0), 1e-12))
        )
        report[f"{mode}_x_scale_eq_recip_mul"] = bool(
            np.array_equal(ts, np.maximum(amax * (np.float32(1) / np.float32(448.0)), 1e-12))
        )

    # The kernel's records, one 128-token page per target page.
    pages = T // 128
    cache = jnp.zeros((pages + 1, 32, 4, 256), jnp.uint8)
    targets = jnp.arange(1, pages + 1, dtype=jnp.int32)
    ang = inp["angle"]
    cos = jnp.asarray(np.cos(ang).astype(ml_dtypes.bfloat16))
    sin = jnp.asarray(np.sin(ang).astype(ml_dtypes.bfloat16))
    got = key_prologue_pallas(
        jnp.asarray(inp["x"]), jnp.asarray(inp["wk"]), jnp.asarray(inp["wk_scale"]),
        jnp.asarray(inp["nw"]), jnp.asarray(inp["nb"]), cos, sin, cache, targets, eps=EPS,
    )
    got = np.asarray(got)[1:].reshape(T, 256)

    for mode in ("eager", "compiled"):
        k = jnp.asarray(tr[f"{mode}_k"].astype(ml_dtypes.bfloat16))
        vals, _, sb = quantize_fp8_ue8m0(k)
        ref = np.zeros((T, 256), np.uint8)
        ref[:, :D] = np.asarray(jax.lax.bitcast_convert_type(vals, jnp.uint8))
        ref[:, D] = np.asarray(sb)[:, 0]
        row_eq = np.all(got == ref, axis=1)
        report[f"vs_{mode}"] = {
            "exact_records": float(row_eq.mean()),
            "byte_mismatch": int((got != ref).sum()),
            "scale_byte_max_diff": int(np.abs(got[:, D].astype(int) - ref[:, D]).max()),
            "bad_rows_head": np.nonzero(~row_eq)[0][:20].tolist(),
        }
    report["eager_vs_compiled_k_equal"] = bool(np.array_equal(tr["eager_k"], tr["compiled_k"]))
    report["stages"] = _stage_diagnostic(inp, tr, cos, sin)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


def _bf16_ulp(a, b):
    """Distance in bf16 ULPs between two arrays already holding bf16 values."""
    import ml_dtypes

    def ordered(x):
        i = np.asarray(x, np.float32).astype(ml_dtypes.bfloat16).view(np.int16).astype(np.int32)
        return np.where(i < 0, -(i & 0x7FFF), i)

    return np.abs(ordered(a) - ordered(b))


def _compare(ours, theirs):
    ulp = _bf16_ulp(ours, theirs)
    return {
        "exact_frac": float(np.mean(ulp == 0)),
        "max_ulp": int(ulp.max()),
        "frac_gt_1ulp": float(np.mean(ulp > 1)),
        "max_abs": float(np.abs(np.asarray(ours, np.float32) - theirs).max()),
    }


def _stage_diagnostic(inp, tr, cos, sin):
    """The kernel's math, one stage at a time in plain JAX, against torch.

    Each stage is fed torch's output of the stage before it, so a mismatch is
    attributed to one stage. ``chained`` runs the whole chain from x.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    f32, bf = jnp.float32, jnp.bfloat16
    rt = lambda a: lax.reduce_precision(a.astype(f32), exponent_bits=8, mantissa_bits=7)

    def proj(x):
        x = x.astype(f32)
        xs = jnp.maximum(jnp.max(jnp.abs(x), axis=1, keepdims=True) / 448.0, 1e-12)
        xq = (x / xs).astype(jnp.float8_e4m3fn)
        acc = lax.dot_general(
            xq, jnp.asarray(inp["wk"]), (((1,), (0,)), ((), ())),
            preferred_element_type=f32,
        )
        return rt(acc * xs * jnp.asarray(inp["wk_scale"])[None])

    def norm(k):
        k = k.astype(f32)
        mean = jnp.mean(k, axis=1, keepdims=True)
        var = jnp.mean(jnp.square(k - mean), axis=1, keepdims=True)
        return rt((k - mean) * lax.rsqrt(var + EPS) * jnp.asarray(inp["nw"]) + jnp.asarray(inp["nb"]))

    def rope(k):
        k = k.astype(f32)
        c, s = cos.astype(f32), sin.astype(f32)
        x1, x2 = k[:, 0:ROPE:2], k[:, 1:ROPE:2]
        o1 = rt(rt(x1 * c) - rt(x2 * s))
        o2 = rt(rt(x2 * c) + rt(x1 * s))
        rot = jnp.stack([o1, o2], axis=-1).reshape(k.shape[0], ROPE)
        return jnp.concatenate([rot, k[:, ROPE:]], axis=1)

    x = jnp.asarray(inp["x"])
    t_proj, t_norm, t_k = (jnp.asarray(tr[f"eager_{n}"]) for n in ("k_proj", "k_norm", "k"))
    out = {
        "proj_from_x": _compare(np.asarray(jax.jit(proj)(x)), tr["eager_k_proj"]),
        "norm_from_torch_proj": _compare(np.asarray(jax.jit(norm)(t_proj)), tr["eager_k_norm"]),
        "rope_from_torch_norm": _compare(np.asarray(jax.jit(rope)(t_norm)), tr["eager_k"]),
        "chained_from_x": _compare(np.asarray(jax.jit(lambda a: rope(norm(proj(a))))(x)), tr["eager_k"]),
    }
    del t_k
    # Same stages, but each stage's output is pulled to the host between stages,
    # so XLA cannot fuse across a stage boundary and drop the bf16 round trip.
    p = np.asarray(jax.jit(proj)(x))
    n = np.asarray(jax.jit(norm)(jnp.asarray(p)))
    out["norm_after_host_proj"] = _compare(n, tr["eager_k_norm"])
    out["norm_fused_from_x"] = _compare(np.asarray(jax.jit(lambda a: norm(proj(a)))(x)), tr["eager_k_norm"])
    out["rope_variants"] = _rope_variants(tr["eager_k_norm"], np.asarray(cos), np.asarray(sin), tr["eager_k"])

    # Which boundary loses rounding once rope joins the jit.
    out["norm_rope_fused_from_torch_proj"] = _compare(
        np.asarray(jax.jit(lambda a: rope(norm(a)))(t_proj)), tr["eager_k"]
    )
    n_host = np.asarray(jax.jit(lambda a: norm(proj(a)))(x))
    out["chain_host_between_norm_rope"] = _compare(
        np.asarray(jax.jit(rope)(jnp.asarray(n_host))), tr["eager_k"]
    )
    # reduce-precision ops XLA keeps in the optimized chained program.
    chained = jax.jit(lambda a: rope(norm(proj(a)))).lower(x).compile().as_text()
    out["chained_hlo_reduce_precision_ops"] = chained.count("reduce-precision(")
    out["chained_hlo_convert_bf16_ops"] = chained.count("bf16[")

    # The kernel's own per-stage values, compiled by Mosaic.
    from sglang_tpu.attention.pallas.dsa_key_prologue import key_stages_pallas

    ks = key_stages_pallas(
        x, jnp.asarray(inp["wk"]), jnp.asarray(inp["wk_scale"]),
        jnp.asarray(inp["nw"]), jnp.asarray(inp["nb"]), cos, sin, eps=EPS,
    )
    xs_k, p_k, n_k, r_k = (np.asarray(a) for a in ks)
    out["kernel_x_scale_eq_torch"] = bool(np.array_equal(xs_k[:, :1], tr["eager_x_scale"]))
    out["kernel_proj"] = _compare(p_k, tr["eager_k_proj"])
    out["kernel_norm"] = _compare(n_k, tr["eager_k_norm"])
    out["kernel_rope"] = _compare(r_k, tr["eager_k"])
    out["proj_probe"] = _proj_probe(inp, x)
    return out


def _proj_probe(inp, x):
    """Split the Mosaic projection: activation quantization vs the FP8 dot."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    from sglang_tpu.attention.pallas.dsa_key_prologue import key_proj_probe_pallas

    f32 = jnp.float32
    wk = jnp.asarray(inp["wk"])

    @jax.jit
    def ref(x):
        xf = x.astype(f32)
        xs = jnp.maximum(jnp.max(jnp.abs(xf), axis=1, keepdims=True) / 448.0, 1e-12)
        xq = (xf / xs).astype(jnp.float8_e4m3fn)
        acc = lax.dot_general(xq, wk, (((1,), (0,)), ((), ())), preferred_element_type=f32)
        return xs, xq, acc

    xs, xq, acc = ref(x)
    inv = jnp.broadcast_to(1.0 / xs, (x.shape[0], 128)).astype(f32)
    xqd, accd, xqm, accm, acce = (
        np.asarray(a) for a in key_proj_probe_pallas(x, wk, inv, xq)
    )
    xq_ref = np.asarray(xq.astype(f32))
    acc_ref = np.asarray(acc)

    def q_cmp(a):
        diff = a != xq_ref
        return {"exact_frac": float(1 - diff.mean()), "mismatch": int(diff.sum()),
                "rows_with_mismatch": int(diff.any(1).sum())}

    def acc_cmp(a):
        d = np.abs(a - acc_ref)
        return {"exact_frac": float(np.mean(d == 0)), "max_abs": float(d.max()),
                "max_rel": float((d / np.maximum(np.abs(acc_ref), 1e-30)).max())}

    from sglang_tpu.attention.pallas.dsa_key_prologue import _DOT_VARIANTS, key_dot_variant_pallas

    # f64 reference of the same FP8 products: which side is closer to exact.
    exact = xq_ref.astype(np.float64) @ np.asarray(wk.astype(f32)).astype(np.float64)

    def err64(a):
        d = np.abs(np.asarray(a, np.float64) - exact)
        return {"max_abs": float(d.max()), "mean_abs": float(d.mean())}

    variants = {}
    for name in _DOT_VARIANTS:
        try:
            a = np.asarray(key_dot_variant_pallas(xq, wk, name))
        except Exception as e:  # a form Mosaic rejects; record it, keep going
            variants[name] = {"error": repr(e)[:300]}
            continue
        variants[name] = {"vs_xla": acc_cmp(a), "vs_f64": err64(a)}
    return {
        "xq_div_vs_jax": q_cmp(xqd),
        "xq_mul_vs_jax": q_cmp(xqm),
        "acc_div_vs_jax": acc_cmp(accd),
        "acc_mul_vs_jax": acc_cmp(accm),
        "acc_given_xq_vs_jax": acc_cmp(acce),
        "xla_vs_f64": err64(acc_ref),
        "dot_variants": variants,
    }


def _rope_variants(k_norm, cos, sin, torch_k):
    """Interleaved rope on torch's LayerNorm output, in NumPy (no XLA rewrites),
    under several rounding rules; each is compared with torch's rope output."""
    import ml_dtypes

    rt = lambda a: np.asarray(a, np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)
    k = np.asarray(k_norm, np.float32)
    c, s = rt(np.asarray(cos, np.float32)), rt(np.asarray(sin, np.float32))
    x1, x2 = k[:, 0:ROPE:2], k[:, 1:ROPE:2]
    f64 = lambda a: a.astype(np.float64)
    variants = {
        # Kernel today: round after each multiply and after the add.
        "round_each_op": (rt(rt(x1 * c) - rt(x2 * s)), rt(rt(x2 * c) + rt(x1 * s))),
        # f32 arithmetic, one round at the end.
        "f32_round_once": (rt(x1 * c - x2 * s), rt(x2 * c + x1 * s)),
        # Exact products and sum (FMA-like), one round at the end.
        "exact_round_once": (
            rt((f64(x1) * f64(c) - f64(x2) * f64(s)).astype(np.float32)),
            rt((f64(x2) * f64(c) + f64(x1) * f64(s)).astype(np.float32)),
        ),
        # One product rounded to f32, the other fused (a*b + c with one round).
        "fma_first_product": (
            rt((f64((x1 * c).astype(np.float32)) - f64(x2) * f64(s)).astype(np.float32)),
            rt((f64((x2 * c).astype(np.float32)) + f64(x1) * f64(s)).astype(np.float32)),
        ),
    }
    out = {}
    for name, (o1, o2) in variants.items():
        got = k.copy()
        got[:, 0:ROPE:2], got[:, 1:ROPE:2] = o1, o2
        res = _compare(got[:, :ROPE], torch_k[:, :ROPE])
        bad = np.nonzero(_bf16_ulp(got[:, :ROPE], torch_k[:, :ROPE]) > 0)
        res["mismatch_head"] = [
            {"tok": int(t), "dim": int(d), "ours": float(got[t, d]), "torch": float(torch_k[t, d]),
             "x1": float(k[t, d - d % 2]), "x2": float(k[t, d - d % 2 + 1]),
             "cos": float(c[t, d // 2]), "sin": float(s[t, d // 2])}
            for t, d in list(zip(*bad))[:5]
        ]
        out[name] = res
    out["non_rope_dims_exact"] = bool(np.array_equal(rt(k[:, ROPE:]), torch_k[:, ROPE:]))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("torch", "jax", "dots"), required=True)
    p.add_argument("--npz", default="/tmp/k_prologue_torch.npz")
    p.add_argument("--out", default="/tmp/k_prologue_compare.json")
    a = p.parse_args()
    if a.stage == "torch":
        stage_torch(a.npz)
    elif a.stage == "jax":
        stage_jax(a.npz, a.out)
    else:
        # Mosaic vs XLA projection probe; JAX only, no torch inputs needed.
        os.environ.setdefault("JAX_PLATFORMS", "tpu")
        import jax.numpy as jnp

        inp = _inputs()
        res = _proj_probe(inp, jnp.asarray(inp["x"]))
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res, indent=1))
