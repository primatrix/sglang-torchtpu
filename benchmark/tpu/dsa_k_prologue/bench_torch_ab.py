"""Test-branch stub: only ``_profile`` from feat/migrate-dsa-kernels' bench_torch_ab.py
(the full file imports indexer ops that are not on main)."""

import os
import sys

import torch


def _profile(fn, args, iters, out_dir):
    """Device time per op of ``iters`` warm calls (xplane_summary), or the error."""
    from pathlib import Path

    from sglang_tpu.profiler import start_tpu_profile, stop_tpu_profile

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from xplane_summary import summarize

    torch.accelerator.synchronize()
    trace = start_tpu_profile(output_dir=Path(out_dir), with_cpu=False)
    try:
        for _ in range(iters):
            fn(*args)
        torch.accelerator.synchronize()
    finally:
        stop_tpu_profile(trace)
    try:
        return summarize(out_dir, iters)
    except Exception as e:  # report, do not hide
        return {"error": repr(e)[:2000]}
