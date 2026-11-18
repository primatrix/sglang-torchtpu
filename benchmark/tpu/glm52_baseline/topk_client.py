"""Test only: drive long extends on attention-DP rank 0 for the top-k stats.

One smoke request (warmup is over and outputs are sane), then ``<stats-dir>/ENABLE``
and ``--requests`` natural-code prompts of ``--prompt-tokens`` each, routed to
DP rank 0 with output 1. The server's sparse MLA backend (dsa_topk_stats.py)
records the extend top-k rows of rank 0. Writes summary.json with PASS when every
request finished and stats were written.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp

from benchmark.tpu.glm52_baseline.client import (
    api,
    freeze_inputs,
    generate,
    load_tokenizer,
    save,
)


async def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        info = await api(session, a.base_url, "/server_info")
        save(a.output / "server-info.json", info)
        tokenizer = await asyncio.to_thread(load_tokenizer, a.model)
        text = tokenizer.apply_chat_template(
            [dict(role="user", content="What is 2 + 2? Answer with the number only.")],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        smoke = await generate(session, a.base_url, [], 0, 16, a.output, "smoke", text=text)
        assert "4" in smoke["text"], smoke["text"]
        await api(session, a.base_url, "/flush_cache")
        _, ids = await asyncio.to_thread(freeze_inputs, a.model, a.source, a.output, "shared")
        (a.stats_dir / "ENABLE").write_text("1")
        rows = []
        for i in range(a.requests):
            prompt = ids[i * a.prompt_tokens : (i + 1) * a.prompt_tokens]
            assert len(prompt) == a.prompt_tokens
            t0 = time.perf_counter()
            r = await generate(session, a.base_url, prompt, 0, 1, a.output, f"long-{i}")
            rows.append(dict(i=i, seconds=round(time.perf_counter() - t0, 2),
                             prompt_tokens=r["meta"]["prompt_tokens"]))
            print(json.dumps(rows[-1]), flush=True)
        (a.stats_dir / "ENABLE").unlink()
        await asyncio.sleep(5)  # last host callbacks
        stats = a.stats_dir / "stats-rank0.jsonl"
        ok = stats.exists() and stats.stat().st_size > 0
        if ok:
            # One line the runner's progress tail shows: mean distinct rows per
            # query group, by layer and group size (deep-context calls).
            agg = {}
            for line in stats.read_text().splitlines():
                rec = json.loads(line)
                for g, d in (rec.get("groups") or {}).items():
                    if d is not None:
                        agg.setdefault(f"L{rec['layer']}/{g}", []).append(d)
            summary = {k: round(sum(v) / len(v), 1) for k, v in sorted(agg.items())}
            save(a.output / "groups.json", summary)
            print("GROUPS " + json.dumps(summary), flush=True)
        save(a.output / "summary.json",
             dict(status="PASS" if ok else "FAIL", requests=rows, stats=str(stats)))


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--model", required=True)
    p.add_argument("--source", type=Path, default=Path.cwd())
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stats-dir", type=Path, required=True)
    p.add_argument("--requests", type=int, default=3)
    p.add_argument("--prompt-tokens", type=int, default=53248)  # 49152 + 4096
    p.add_argument("--decode-prefix", default="shared")  # accepted, unused
    return p.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
