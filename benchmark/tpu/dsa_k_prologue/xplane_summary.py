"""Per-op device time from an xprof capture (``*.xplane.pb``).

Sums the duration of every event on the TPU planes' "XLA Ops" line by op name
(and on "XLA Modules" by program name), divided by the number of captured
calls. Only device time; host time is not read.

No profiler package is needed: the few XSpace fields read here are decoded
from the protobuf wire format directly (field numbers from tsl's xplane.proto:
XSpace.planes 1; XPlane name 2, lines 3, event_metadata 4 (map: key 1, value
2); XLine name 2, events 4; XEvent metadata_id 1, duration_ps 3;
XEventMetadata name 2). The plane and line names found are reported, so a
wrong decode shows up as unexpected names rather than plausible numbers.
"""

import collections
import glob
import os


def _varint(buf, i):
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _fields(buf):
    """Yield ``(field_number, value)``: int for varints, bytes for length-delimited."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 2:
            size, i = _varint(buf, i)
            value, i = bytes(buf[i : i + size]), i + size
        elif wire == 1:
            value, i = None, i + 8
        elif wire == 5:
            value, i = None, i + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield field, value


def _first(buf, field, default):
    for f, v in _fields(buf):
        if f == field:
            return v
    return default


def _planes(data):
    """``(plane_name, {metadata_id: name}, [(line_name, [(metadata_id, duration_ps)])])``."""
    for f, plane in _fields(data):
        if f != 1:
            continue
        name, names, lines = "", {}, []
        for pf, pv in _fields(plane):
            if pf == 2:
                name = pv.decode("utf-8", "replace")
            elif pf == 4:  # map<int64, XEventMetadata> entry
                key, meta = 0, b""
                for ef, ev in _fields(pv):
                    if ef == 1:
                        key = ev
                    elif ef == 2:
                        meta = ev
                names[key] = _first(meta, 2, b"?").decode("utf-8", "replace")
            elif pf == 3:
                line_name, events = "", []
                for lf, lv in _fields(pv):
                    if lf == 2:
                        line_name = lv.decode("utf-8", "replace")
                    elif lf == 4:
                        mid = dur = 0
                        for xf, xv in _fields(lv):
                            if xf == 1:
                                mid = xv
                            elif xf == 3:
                                dur = xv
                        events.append((mid, dur))
                lines.append((line_name, events))
        yield name, names, lines


def summarize(trace_dir: str, calls: int, top: int = 25) -> dict:
    """``{"ops": [(name, us_per_call), ...], "modules": [...], "device_us": total}``."""
    files = glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"), recursive=True)
    if not files:
        return {"error": f"no .xplane.pb under {trace_dir}"}
    ops, modules = collections.Counter(), collections.Counter()
    planes, line_names = set(), set()
    for path in files:
        with open(path, "rb") as f:
            data = memoryview(f.read())
        for plane_name, names, lines in _planes(data):
            planes.add(plane_name)
            if not plane_name.startswith("/device:TPU"):
                continue
            for line_name, events in lines:
                line_names.add(line_name)
                target = {"XLA Ops": ops, "XLA Modules": modules}.get(line_name)
                if target is None:
                    continue
                for mid, dur in events:
                    target[names.get(mid, "?")] += dur
    per_call = lambda c: [(n, ps / 1e6 / calls) for n, ps in c.most_common(top)]
    return {
        "files": len(files),
        "planes": sorted(planes),
        "tpu_lines": sorted(line_names),
        "device_us": sum(ops.values()) / 1e6 / calls,
        "ops": per_call(ops),
        "modules": per_call(modules),
    }
