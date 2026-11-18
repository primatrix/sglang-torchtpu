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
import collections
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
        # Window stats (no dropped time). The runner's log tail cuts lines at
        # 300 chars, so one short line per number group; the full rows go to a file.
        try:
            rows = windowed(f)
        except Exception as e:  # report, do not hide
            rows = []
            print("WIN " + json.dumps({"error": repr(e)[:250]}), flush=True)
        tag = f.split("-TP-")[-1].split("/")[0][:20]
        with open(f"{out_dir}/window-{tag}.json", "w") as fh:
            json.dump(rows, fh)
        for r in rows:
            p = r["plane"][-6:]
            print("WIN " + json.dumps({k: r[k] for k in (
                "window_us", "window_frac_of_module", "tc_busy_us", "tc_idle_us",
                "sc_busy_us", "tc_idle_while_sc_busy_us", "attn_kernels_in_window")}
                | {"r": tag, "p": p}), flush=True)
            print("WINC " + json.dumps({"r": tag, "p": p, "c": r["categories"]})[:290],
                  flush=True)
            for i in range(0, 15, 3):
                print("WINO " + json.dumps({"r": tag, "p": p,
                                            "o": r["other_top"][i:i + 3]})[:290], flush=True)
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


def _abs_lines(path):
    """``[(plane, line, [(name, start_ps, end_ps)])]`` with absolute line times.

    XLine.timestamp_ns (field 3) is each line's base; event offsets are relative
    to it, so lines are put on one clock before comparing them.
    """
    from xplane_summary import _fields

    data = memoryview(open(path, "rb").read())
    out = []
    for f, plane in _fields(data):
        if f != 1:
            continue
        pname, names, raw = "", {}, []
        for pf, pv in _fields(plane):
            if pf == 2:
                pname = pv.decode("utf-8", "replace")
            elif pf == 4:
                key, meta = 0, b""
                for ef, ev in _fields(pv):
                    if ef == 1:
                        key = ev
                    elif ef == 2:
                        meta = ev
                nm = b"?"
                for mf, mv in _fields(meta):
                    if mf == 2:
                        nm = mv
                        break
                names[key] = nm.decode("utf-8", "replace")
            elif pf == 3:
                lname, ts, evs = "", 0, []
                for lf, lv in _fields(pv):
                    if lf == 2:
                        lname = lv.decode("utf-8", "replace")
                    elif lf == 3:
                        ts = lv
                    elif lf == 4:
                        mid = off = dur = 0
                        for xf, xv in _fields(lv):
                            if xf == 1:
                                mid = xv
                            elif xf == 2:
                                off = xv
                            elif xf == 3:
                                dur = xv
                        evs.append((mid, off, dur))
                raw.append((lname, ts, evs))
        if not pname.startswith("/device:TPU"):
            continue
        for lname, ts, evs in raw:
            base = ts * 1000
            out.append((pname, lname, [(names.get(m, "?"), base + o, base + o + d)
                                       for m, o, d in evs]))
    return out


def _union(iv):
    tot, end = 0, None
    for s, e in sorted(iv):
        if end is None or s > end:
            tot += e - s
            end = e
        elif e > end:
            tot += e - end
            end = e
    return tot


def _clip(iv, w0, w1):
    return [(max(s, w0), min(e, w1)) for s, e in iv if e > w0 and s < w1]


def windowed(path):
    """Statistics over the part of one device file that every line still covers.

    A capture over the size limit drops the tail, and dropped time reads as idle.
    Window: from the first module start to the earliest "last event end" over the
    TensorCore op line and every SparseCore line, so nothing in it was dropped.
    Inside it: TensorCore op busy / idle, SparseCore busy (union of all SC lines),
    TC idle while SC is busy (waiting on the gather), op self time by category,
    and the largest uncategorised ops by name.
    """
    lines = _abs_lines(path)
    tc_planes = sorted({p for p, l, ev in lines if l == "XLA Ops" and ev})
    rows = []
    for tp in tc_planes:
        ops = [x for p, l, ev in lines if p == tp and l == "XLA Ops" for x in ev]
        mods = [x for p, l, ev in lines if p == tp and l == "XLA Modules" for x in ev]
        # SparseCore planes on this chip: any SC plane (they are per chip, not per TC).
        sc_lines = [(p, l, ev) for p, l, ev in lines if "SparseCore" in p and ev]
        if not ops or not mods:
            continue
        w0 = min(s for _, s, _ in mods)
        ends = [max(e for _, _, e in ops)] + [max(e for _, _, e in ev) for _, _, ev in sc_lines]
        w1 = min(ends)
        if w1 <= w0:
            continue
        op_iv = _clip([(s, e) for _, s, e in ops], w0, w1)
        sc_iv = _clip([(s, e) for _, _, ev in sc_lines for _, s, e in ev], w0, w1)
        win = w1 - w0
        tc_busy = _union(op_iv)
        sc_busy = _union(sc_iv)
        both = tc_busy + sc_busy - _union(op_iv + sc_iv)
        sc_only = sc_busy - both  # SC busy, TC op line empty
        # Self time by name inside the window (top-level approximation: events
        # minus nested children on the TC op line).
        evs = sorted(((n, max(s, w0), min(e, w1)) for n, s, e in ops if e > w0 and s < w1),
                     key=lambda x: (x[1], -(x[2] - x[1])))
        self_t = collections.Counter()
        stack = []
        for n, s, e in evs:
            while stack and stack[-1][2] <= s:
                stack.pop()
            if stack:
                self_t[stack[-1][0]] -= min(e, stack[-1][2]) - s
            self_t[n] += e - s
            stack.append((n, s, e))
        cats, other = collections.Counter(), collections.Counter()
        for n, t in self_t.items():
            low = n.lower()
            lab = next((lb for lb, subs in CATEGORIES if any(x.lower() in low for x in subs)),
                       "other")
            cats[lab] += t
            if lab == "other":
                other[n[:60]] += t
        us = lambda ps: round(ps / 1e6, 1)
        n_attn = sum(1 for n, s, e in ops if "MLA-" in n and w0 <= s < w1)
        rows.append(dict(
            plane=tp, window_us=us(win), window_frac_of_module=round(
                win / max(1, max(e for _, _, e in mods) - w0), 3),
            tc_busy_us=us(tc_busy), tc_idle_us=us(win - tc_busy),
            sc_busy_us=us(sc_busy), tc_idle_while_sc_busy_us=us(sc_only),
            attn_kernels_in_window=n_attn,
            categories={k: us(v) for k, v in cats.most_common()},
            other_top=[(k, us(v)) for k, v in other.most_common(15)],
        ))
    return rows


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
