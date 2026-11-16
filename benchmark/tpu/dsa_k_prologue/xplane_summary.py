"""Per-op device time from an xprof capture (``*.xplane.pb``).

Sums the duration of every event on the TPU planes' "XLA Ops" line by op name
(and on "XLA Modules" by program name), divided by the number of captured
calls. Only device time; host time is not read.

The XPlane protobuf ships with several packages; the first that imports wins.
"""

import collections
import glob
import os


def _xplane_pb2():
    errors = []
    for mod in (
        "xprof.protobuf.xplane_pb2",
        "tensorboard_plugin_profile.protobuf.xplane_pb2",
        "tsl.profiler.protobuf.xplane_pb2",
        "tensorflow.tsl.profiler.protobuf.xplane_pb2",
    ):
        try:
            return __import__(mod, fromlist=["XSpace"])
        except Exception as e:  # report every attempt if none works
            errors.append(f"{mod}: {e!r}")
    raise ImportError("no xplane_pb2 found:\n" + "\n".join(errors))


def summarize(trace_dir: str, calls: int, top: int = 25) -> dict:
    """``{"ops": [(name, us_per_call), ...], "modules": [...], "device_us": total}``."""
    files = glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"), recursive=True)
    if not files:
        return {"error": f"no .xplane.pb under {trace_dir}"}
    pb2 = _xplane_pb2()
    ops, modules = collections.Counter(), collections.Counter()
    planes = []
    for path in files:
        space = pb2.XSpace()
        with open(path, "rb") as f:
            space.ParseFromString(f.read())
        for plane in space.planes:
            if not plane.name.startswith("/device:TPU"):
                continue
            planes.append(plane.name)
            names = {k: v.name for k, v in plane.event_metadata.items()}
            for line in plane.lines:
                target = {"XLA Ops": ops, "XLA Modules": modules}.get(line.name)
                if target is None:
                    continue
                for ev in line.events:
                    target[names.get(ev.metadata_id, "?")] += ev.duration_ps
    per_call = lambda c: [(n, ps / 1e6 / calls) for n, ps in c.most_common(top)]
    return {
        "planes": sorted(set(planes)),
        "device_us": sum(ops.values()) / 1e6 / calls,
        "ops": per_call(ops),
        "modules": per_call(modules),
    }
