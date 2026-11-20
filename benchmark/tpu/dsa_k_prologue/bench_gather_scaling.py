"""Test only: why Source's 256-query gather is not 4x Target's 64-query gather.

extend_c1 top-k (physical slots), page 64, FP8 KV, single chip, JAX. Each row is
one jitted SparseCore gather of ``n`` queries x 2048 rows, timed as the device
"XLA Modules" duration (median of several captured calls), so chunk size and
gather kind are varied one at a time:

* ``src_latent``: Source ``_sparse_core_gather`` of the latent rows (512 B/row).
* ``src_rope``: Source ``_sparse_core_gather_rope`` (grouped [S/4, 4, 128] RoPE).
* ``src_split``: both, as ``_gather_split_fp8`` (what Source runs).
* ``tgt``: Target ``dsa_gather`` (NoPE + RoPE in one kernel).

Also prints, for ``src_split`` at 256 and ``tgt`` at 64, every SparseCore-plane
event name with its start/end, so each SC call is identified by name, not by
duration. Prints ``GSC`` lines.
"""

import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "tpu")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_fp8_attention_ab as att  # noqa: E402
from check_physical_topk import to_physical  # noqa: E402
from xplane_timeline import events  # noqa: E402

FP = att.FP = 64
SIZES = (64, 128, 256)


def emit(**kw):
    print("GSC " + json.dumps(kw), flush=True)


def main():
    import jax
    import jax.numpy as jnp

    from sglang_tpu.attention.pallas.mla_sparse import dsa_gather
    from sglang_tpu.attention.pallas.source_dsa.kernels.dsa import gather as sg
    from sglang_tpu.attention.pallas.source_dsa.kernels.dsa.fp8_kv_cache import SplitFP8KVCache

    c = att.case("pr2198_extend_c1")
    P, K = c["P"], c["topk"].shape[1]
    slots = jnp.asarray(to_physical(c["topk"], c["pi"], c["ppr"], c["cu_q"], page=FP))
    lat = jnp.zeros((P * FP, 4, 128), jnp.uint8)
    rope = jnp.zeros((P * FP // 4, 4, 128), jnp.uint8)
    no = jnp.zeros((P, FP, 4, 128), jnp.uint8)
    ro = jnp.zeros((P, FP // 4, 4, 128), jnp.uint8)
    out_dir = "/tmp/gsc"

    def module_us(tag, jf, args, dump_sc=False, calls=5):
        for _ in range(3):
            jax.block_until_ready(jf(*args))
        d = os.path.join(out_dir, tag)
        jax.profiler.start_trace(d)
        try:
            for _ in range(calls):
                jax.block_until_ready(jf(*args))
        finally:
            jax.profiler.stop_trace()
        ev = events(d)
        mods = sorted((s, e) for p, ln, n, s, e in ev
                      if " SparseCore" not in p and ln == "XLA Modules" and f"jit_{tag}" in n)
        durs = sorted((e - s) / 1e6 for s, e in mods)
        r = {"module_us": round(durs[len(durs) // 2], 1) if durs else None,
             "n_modules": len(durs)}
        if dump_sc and mods:
            m0, m1 = mods[len(mods) // 2]
            sc = {}
            for p, ln, n, s, e in ev:
                if " SparseCore" in p and not ln.startswith("TEC") and e > m0 and s < m1:
                    sc.setdefault((ln, n[:80]), []).append(
                        [round((s - m0) / 1e6, 1), round((e - m0) / 1e6, 1)])
            r["sc_events"] = [{"line": ln, "name": n, "iv": iv[:20]}
                              for (ln, n), iv in sorted(sc.items())][:40]
        return r

    for n in SIZES:
        idx = slots[:n].reshape(-1)
        idx2 = slots[:n]
        variants = {
            "src_latent": (lambda t, i: sg._sparse_core_gather(t, i, skip_invalid=True),
                           (lat, idx)),
            "src_rope": (lambda t, i: sg._sparse_core_gather_rope(t, i), (rope, idx)),
            "src_split": (lambda a, b, i: sg._gather_split_fp8(SplitFP8KVCache(a, b), i),
                          (lat, rope, idx)),
            "tgt": (lambda a, b, i: dsa_gather.dsa_gather(
                a, b, i.reshape(-1), num_valid_indices=jnp.int32(i.size)), (no, ro, idx2)),
        }
        for kind, (fn, args) in variants.items():
            tag = f"{kind}_{n}"
            r = {"kind": kind, "queries": n, "rows": n * K}
            try:
                def named(*xs, _fn=fn):
                    return _fn(*xs)

                named.__name__ = tag
                jf = jax.jit(named)
                dump = (kind == "src_split" and n == 256) or (kind == "tgt" and n == 64)
                r.update(module_us(tag, jf, args, dump_sc=dump))
                if r["module_us"]:
                    r["ns_per_row"] = round(r["module_us"] * 1e3 / (n * K), 3)
                    r["us_per_query"] = round(r["module_us"] / n, 3)
            except Exception as e:  # report, do not hide; the other rows still run
                r["fail"] = repr(e)[:800]
            emit(**r)
    print("GSC_DONE", flush=True)


if __name__ == "__main__":
    main()
