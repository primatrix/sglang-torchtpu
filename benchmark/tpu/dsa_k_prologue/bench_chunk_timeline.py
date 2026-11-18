"""Test only: do adjacent chunks' SC gather and TC attention already overlap?

Runs bench_target_gather_sweep for extend_c1 (``target`` and the SparseCore
layout at chunk 128 / batch 16 / atoms 16 / overlap_rope), keeps its xprof
traces, then reads one call's device timeline from the ``.xplane.pb`` files:

* every TPU line's events in the second captured call (name, start, duration),
  printed compactly as ``TL`` lines so the timeline can be read from the log;
* ``OVL``: per variant, the attention intervals (``MLA-*``) and gather intervals
  (``native_sc`` / ``dsa_gather`` / ``sparse-core``), the union time of each,
  and how much gather time falls inside an attention interval of a *different*
  chunk (the overlap a CSA2-style pipeline would add).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import glob  # noqa: E402

XP = "/tmp/chunk_tl"
VAR = "sc_phys_c128_b16"
ATT = ("MLA-tensorcore", "MLA-sparsecore")
GAT = ("native_sc", "dsa_gather", "sparse-core", "sc_dsa_gather")


def union(iv):
    tot, end = 0, None
    for s, e in sorted(iv):
        if end is None or s > end:
            tot += e - s
            end = e
        elif e > end:
            tot += e - end
            end = e
    return tot


def inter(a, b):
    """Total length of union(a) intersected with union(b)."""
    return union(a) + union(b) - union(a + b)


def analyse(tag, d):
    from xplane_summary import _planes

    files = glob.glob(os.path.join(d, "**", "*.xplane.pb"), recursive=True)
    if not files:
        print("OVL " + json.dumps({"variant": tag, "error": f"no trace in {d}"}), flush=True)
        return
    lines = []  # (plane, line, [(name, start_ps, end_ps)])
    for path in files:
        data = memoryview(open(path, "rb").read())
        for pname, names, plines in _planes(data):
            if not pname.startswith("/device:TPU"):
                continue
            for lname, evs in plines:
                lines.append((pname, lname, [(names.get(m, "?"), o, o + du) for m, o, du in evs]))
    # The second call: the second "XLA Modules" event on the first TPU plane.
    mods = sorted(e for p, l, ev in lines if l == "XLA Modules" for e in ev
                  if p == min(pp for pp, ll, _ in lines if ll == "XLA Modules"))
    mods = [m for m in mods if (m[2] - m[1]) > 1e6]  # > 1 us
    if len(mods) < 2:
        print("OVL " + json.dumps({"variant": tag, "error": f"{len(mods)} module events"}),
              flush=True)
        return
    w0, w1 = mods[1][1], mods[1][2]
    att, gat = [], []
    printed = 0
    for pname, lname, evs in lines:
        sel = [(n, s, e) for n, s, e in evs if s < w1 and e > w0]
        if not sel:
            continue
        # Compact timeline (capped so the log stays readable): TensorCore lines
        # and the first SparseCore lines, us relative to the call start.
        budget = 0 if printed >= 500 else 60
        for n, s, e in sorted(sel, key=lambda x: x[1])[:budget]:
            printed += 1
            print("TL " + json.dumps({"v": tag, "p": pname[-12:], "l": lname[:24], "n": n[:48],
                                      "t": round((s - w0) / 1e6, 1),
                                      "d": round((e - s) / 1e6, 1)}), flush=True)
        if lname in ("XLA Modules", "Steps"):
            continue
        for n, s, e in sel:
            if any(k in n for k in ATT):
                att.append((max(s, w0), min(e, w1)))
            elif any(k in n for k in GAT):
                gat.append((max(s, w0), min(e, w1), pname, lname))
    gi = [(s, e) for s, e, _, _ in gat]
    att_sorted = sorted(att)
    # Gather time inside some attention interval (any chunk) vs total gather.
    row = {
        "variant": tag,
        "call_us": round((w1 - w0) / 1e6, 1),
        "attention_n": len(att), "attention_union_us": round(union(att) / 1e6, 1),
        "gather_n": len(gat), "gather_union_us": round(union(gi) / 1e6, 1),
        "gather_lines": sorted({f"{p[-12:]}|{l[:24]}" for _, _, p, l in gat})[:8],
        "gather_inside_attention_us": round(inter(gi, att) / 1e6, 1),
        "neither_us": round(((w1 - w0) - union(gi + att)) / 1e6, 1),
        "attention_first_us": [round((s - w0) / 1e6, 1) for s, _ in att_sorted[:20]],
    }
    print("OVL " + json.dumps(row), flush=True)


def main():
    import bench_target_gather_sweep as sw

    sys.argv = ["sweep", "--cases", "pr2198_extend_c1", "--only", VAR,
                "--iters", "3", "--repeats", "3", "--xprof-dir", XP]
    sw.main()
    for tag in ("target", VAR):
        analyse(tag, f"{XP}/sweep_pr2198_extend_c1_{tag}")
    print("OVL_DONE", flush=True)


if __name__ == "__main__":
    main()
