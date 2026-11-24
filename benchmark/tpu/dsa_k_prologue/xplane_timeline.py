"""Test only: absolute-time event intervals from an xprof capture (``*.xplane.pb``).

xplane_summary.py adds durations per line; this keeps each event's absolute
[start, end) (XLine.timestamp_ns * 1000 + XEvent.offset_ps), so TensorCore and
SparseCore planes share one time axis and overlap can be measured directly.

Field numbers (tsl xplane.proto): XSpace.planes 1; XPlane name 2, lines 3,
event_metadata 4; XLine name 2, timestamp_ns 3, events 4; XEvent metadata_id 1,
offset_ps 2, duration_ps 3; XEventMetadata name 2.
"""

import glob
import os

from xplane_summary import _fields, _first


def events(trace_dir, device="/device:TPU:0"):
    """``[(plane, line, name, start_ps, end_ps)]`` for planes of ``device``."""
    out = []
    for path in glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"), recursive=True):
        with open(path, "rb") as f:
            data = memoryview(f.read())
        for fnum, plane in _fields(data):
            if fnum != 1:
                continue
            pname, names, lines = "", {}, []
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
                    names[key] = _first(meta, 2, b"?").decode("utf-8", "replace")
                elif pf == 3:
                    lines.append(pv)
            if pname != device and not pname.startswith(device + " "):
                continue
            for lv in lines:
                lname, ts, evs = "", 0, []
                for lf, x in _fields(lv):
                    if lf == 2:
                        lname = x.decode("utf-8", "replace")
                    elif lf == 3:
                        ts = x
                    elif lf == 4:
                        mid = off = dur = 0
                        for xf, xv in _fields(x):
                            if xf == 1:
                                mid = xv
                            elif xf == 2:
                                off = xv
                            elif xf == 3:
                                dur = xv
                        evs.append((mid, off, dur))
                base = ts * 1000
                for mid, off, dur in evs:
                    out.append((pname, lname, names.get(mid, "?"), base + off,
                                base + off + dur))
    return out


def merge(iv, gap_ps=0):
    """Union of ``[(s, e)]``, joining pieces closer than ``gap_ps``."""
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1] + gap_ps:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def total(iv):
    return sum(e - s for s, e in iv)


def intersect(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            out.append([s, e])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def clip(iv, lo, hi):
    return [[max(s, lo), min(e, hi)] for s, e in iv if e > lo and s < hi]


def call_timeline(trace_dir, call=1, attn_key="MLA-", gather_keys=("native_sc_gather",)):
    """One captured call (``call``-th "XLA Modules" event): per-chunk TC attention
    intervals, SparseCore busy (union of TEC lines), their overlap and the TC
    gaps, all in us from the module start."""
    ev = events(trace_dir)
    mods = sorted((s, e, n) for p, ln, n, s, e in ev
                  if " SparseCore" not in p and ln == "XLA Modules")
    if len(mods) <= call:
        return {"error": f"{len(mods)} module events, wanted call {call}"}
    m0, m1, mname = mods[call]
    us = lambda ps: round(ps / 1e6, 1)  # noqa: E731
    rel = lambda iv: [[us(s - m0), us(e - m0)] for s, e in iv]  # noqa: E731

    tc_ops = [(n, s, e) for p, ln, n, s, e in ev
              if " SparseCore" not in p and ln == "XLA Ops" and e > m0 and s < m1]
    attn = merge([[max(s, m0), min(e, m1)] for n, s, e in tc_ops if attn_key in n])
    tc_gather = merge([[max(s, m0), min(e, m1)] for n, s, e in tc_ops
                       if any(k in n for k in gather_keys)])
    tc_busy = merge([[max(s, m0), min(e, m1)] for _, s, e in tc_ops])
    sc_lines = {}
    for p, ln, n, s, e in ev:
        if " SparseCore" in p and e > m0 and s < m1:
            kind = "TEC" if ln.startswith("TEC") else ln
            sc_lines.setdefault(kind, []).append([max(s, m0), min(e, m1)])
    sc_busy = merge(sc_lines.get("TEC", []), gap_ps=2_000_000)  # join < 2 us gaps
    # Gaps on the TC between attention chunks, and what the SC did in them.
    gaps = [[a[1], b[0]] for a, b in zip(attn, attn[1:]) if b[0] > a[1]]
    lead = [[m0, attn[0][0]]] if attn else []
    tail = [[attn[-1][1], m1]] if attn else []
    other_names = {}
    for n, s, e in tc_ops:
        if attn_key in n or any(k in n for k in gather_keys):
            continue
        other_names[n[:60]] = other_names.get(n[:60], 0) + e - s
    return {
        "module": mname[:60], "module_us": us(m1 - m0), "n_modules": len(mods),
        "attn_chunks": rel(attn), "attn_busy_us": us(total(attn)),
        "tc_gather_ops": rel(tc_gather)[:20],
        "tc_busy_us": us(total(tc_busy)),
        "sc_busy": rel(sc_busy)[:40], "sc_busy_us": us(total(sc_busy)),
        "sc_lines_us": {k: us(total(merge(v))) for k, v in sc_lines.items()},
        "attn_and_sc_us": us(total(intersect(attn, sc_busy))),
        "lead_us": us(total(lead)), "tail_us": us(total(tail)),
        "gaps_between_chunks": rel(gaps), "gaps_us": us(total(gaps)),
        "sc_busy_in_gaps_us": us(total(intersect(merge(gaps + lead), sc_busy))),
        "attn_without_sc_us": us(total(attn) - total(intersect(attn, sc_busy))),
        "tc_other_top": sorted(((k, us(v)) for k, v in other_names.items()),
                               key=lambda x: -x[1])[:8],
    }


def call_events(trace_dir, call=1, min_us=1.0):
    """Raw intervals of one captured call, in us from its module start.

    TensorCore: every "XLA Ops" event of at least ``min_us`` as (name, s, e).
    SparseCore: per non-TEC line its named events, and the TEC lines' union
    (pieces closer than 2 us joined) as ``("TEC", s, e)``. Returns
    ``{"module_us", "tc": [...], "sc": [...], "dropped_small_us"}``.
    """
    ev = events(trace_dir)
    mods = sorted((s, e) for p, ln, n, s, e in ev
                  if " SparseCore" not in p and ln == "XLA Modules")
    if len(mods) <= call:
        return {"error": f"{len(mods)} module events, wanted call {call}"}
    m0, m1 = mods[call]
    r = lambda ps: round((ps - m0) / 1e6, 1)  # noqa: E731
    tc, sc, small, tec = [], [], 0, []
    for p, ln, n, s, e in ev:
        if e <= m0 or s >= m1:
            continue
        s, e = max(s, m0), min(e, m1)
        if " SparseCore" in p:
            if ln.startswith("TEC"):
                tec.append([s, e])
            elif (e - s) >= min_us * 1e6:
                sc.append((f"{ln[:12]}|{n[:48]}", r(s), r(e)))
        elif ln == "XLA Ops":
            if (e - s) >= min_us * 1e6:
                tc.append((n[:56], r(s), r(e)))
            else:
                small += e - s
    sc += [("TEC", r(s), r(e)) for s, e in merge(tec, gap_ps=2_000_000)]
    return {"module_us": round((m1 - m0) / 1e6, 1), "n_modules": len(mods),
            "tc": sorted(tc, key=lambda x: x[1]), "sc": sorted(sc, key=lambda x: x[1]),
            "dropped_small_us": round(small / 1e6, 1)}
