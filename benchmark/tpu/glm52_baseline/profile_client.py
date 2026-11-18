"""Test only: one xprof capture of a GLM-5.2 extend step, split by op category.

Question: in one extend step (2048 new tokens, ~49K prefix), how long does the
TensorCore sit idle while the SparseCore gathers top-k KV rows, and how much
other TensorCore work (MoE, dense projections) is there that a two-micro-batch
overlap could put under that gather?

Flow: smoke request; one untraced 53,248-token request on DP rank 0 (warms every
extend graph); flush; ``/start_profile`` for ``--steps`` forward steps; a second
53,248-token request with different tokens. The capture lands on the extend
chunks at the start of that request (prefix 0-4K, shallower than the 49K case);
``--skip-chunks`` sends that many 2048-token chunks first as a cached prefix so the
captured chunk has a deep prefix. Then every device ``.xplane.pb`` is summarised
with ``xplane_summary.breakdown``: ``PROF`` lines per file.
"""

import argparse
import asyncio
import glob
import json
import os
import time
from pathlib import Path

import aiohttp

from benchmark.tpu.glm52_baseline.client import api, freeze_inputs, generate, load_tokenizer, save

CATEGORIES = [
    ("attn_kernel", ("MLA-tensorcore", "MLA-sparsecore", "MLA-")),
    ("sc_gather", ("native_sc", "dsa_gather", "sparse-core-data-format")),
    ("indexer", ("indexer", "index_topk", "topk", "dsa_bf16")),
    ("moe", ("moe", "gmm", "expert", "all-to-all", "ragged")),
    ("collective", ("all-reduce", "all-gather", "reduce-scatter", "collective", "send", "recv")),
    ("matmul", ("dot", "convolution", "matmul")),
    ("kv_store", ("scatter",)),
]


def summarise(out_dir):
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from xplane_summary import breakdown

    files = sorted(glob.glob(f"{out_dir}/**/*.xplane.pb", recursive=True))
    print("PROF " + json.dumps({"files": len(files),
                                "sizes_mb": [round(os.path.getsize(f) / 2**20) for f in files]}),
          flush=True)
    for f in files:
        dropped = b"Trace Buffers Dropped" in open(f, "rb").read()
        d = os.path.dirname(f)
        try:
            b = breakdown(d, 1, CATEGORIES)
        except Exception as e:  # report, do not hide
            print("PROF " + json.dumps({"file": f[-90:], "error": repr(e)[:400]}), flush=True)
            continue
        if b.get("module_span_us", 0) <= 0:
            continue  # host-only file
        lines = b.get("lines", {})
        sc = {k: v for k, v in lines.items() if "SparseCore" in k}
        tc = {k: v for k, v in lines.items() if "SparseCore" not in k}
        print("PROF " + json.dumps({
            "file": f[-90:], "dropped": dropped,
            "module_span_us": b["module_span_us"], "tc_ops_busy_us": b["ops_busy_us"],
            "tc_idle_in_modules_us": round(b["module_span_us"] - b["ops_busy_us"], 1),
            "categories": b["categories"],
            "other_top": b["other_top"][:8],
            "sc_line_max_us": max(sc.values()) if sc else None,
            "sc_line_median_us": sorted(sc.values())[len(sc) // 2] if sc else None,
            "tc_lines": dict(list(tc.items())[:6]),
        }), flush=True)


async def post_text(session, base, endpoint, payload=None):
    """/start_profile and /stop_profile answer in plain text, not JSON."""
    async with session.post(base + endpoint, json=payload or {}) as r:
        return r.status, (await r.text())[:500]


async def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    prof = (a.output / "xprof").resolve()
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # A capture armed by an earlier attempt would trace the smoke request.
        print("PROF " + json.dumps({"stop_previous": await post_text(
            session, a.base_url, "/stop_profile")}), flush=True)
        save(a.output / "server-info.json", await api(session, a.base_url, "/server_info"))
        tok = await asyncio.to_thread(load_tokenizer, a.model)
        text = tok.apply_chat_template(
            [dict(role="user", content="What is 2 + 2? Answer with the number only.")],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        smoke = await generate(session, a.base_url, [], 0, 16, a.output, "smoke", text=text)
        assert "4" in smoke["text"], smoke["text"]
        _, ids = await asyncio.to_thread(freeze_inputs, a.model, a.source, a.output, "shared")
        n = a.prompt_tokens
        t0 = time.perf_counter()
        await generate(session, a.base_url, ids[:n], 0, 1, a.output, "warm")
        print("PROF " + json.dumps({"warm_s": round(time.perf_counter() - t0, 1)}), flush=True)
        await api(session, a.base_url, "/flush_cache")
        target = ids[n:2 * n]
        assert len(target) == n
        skip = a.skip_chunks * 2048
        if skip:
            # Cached prefix so the traced chunk attends a deep context.
            await generate(session, a.base_url, target[:skip], 0, 1, a.output, "prefix")
        st = await post_text(session, a.base_url, "/start_profile",
                             dict(output_dir=str(prof), num_steps=a.steps,
                                  profile_prefix="extend"))
        print("PROF " + json.dumps({"start_profile": st}), flush=True)
        assert st[0] == 200, st
        t0 = time.perf_counter()
        await generate(session, a.base_url, target[:skip + a.traced_tokens], 0, 1,
                       a.output, "traced")
        print("PROF " + json.dumps({"traced_s": round(time.perf_counter() - t0, 1)}), flush=True)
        # The capture is written after the request; wait for files to settle.
        last, stable = -1, 0
        for _ in range(120):
            await asyncio.sleep(5)
            sz = sum(os.path.getsize(f) for f in glob.glob(f"{prof}/**/*.xplane.pb",
                                                           recursive=True))
            stable = stable + 1 if sz == last and sz > 0 else 0
            last = sz
            if stable >= 3:
                break
    await asyncio.to_thread(summarise, prof)
    save(a.output / "summary.json", dict(status="PASS"))
    print("PROF_DONE", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--model", required=True)
    p.add_argument("--source", type=Path, default=Path.cwd())
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompt-tokens", type=int, default=53248)
    p.add_argument("--skip-chunks", type=int, default=24)  # 49,152-token cached prefix
    p.add_argument("--traced-tokens", type=int, default=2048)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--stats-dir", type=Path, default=None)  # accepted, unused
    p.add_argument("--decode-prefix", default="shared")  # accepted, unused
    return p.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
