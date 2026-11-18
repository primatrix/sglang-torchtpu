"""Test-only: per-chunk duplicate rate of the DSA indexer's extend top-k rows.

Enabled by ``SGLANG_TPU_DSA_TOPK_STATS_DIR``. ``_forward_mla`` registers each
extend step's ``topk_indices`` with ``record``; under ``--tpu-compile-tracer
direct`` the tensor becomes an extra output of the compiled step and the
callback runs on the host after it executes. Nothing is recorded until the
client creates ``<dir>/ENABLE`` (warmup and smoke requests are skipped), and
only global rank 0 (attention-DP rank 0) writes.

Per call, one JSON line in ``stats-rank0.jsonl``: for every block of
``BLOCK`` query rows (the sparse kernel's gather chunk), the valid entries,
the distinct rows among them, and the mean overlap of each query's row set with
the previous query's. The first ``SGLANG_TPU_DSA_TOPK_RAW_LIMIT`` calls whose rows reach position
32768 are also saved raw (``raw-*.npy``) for offline checks.
"""

import json
import os
import threading

import numpy as np
import torch

BLOCK = 128
GROUPS = (4, 8, 16, 32, 64, 128)
_lock = threading.Lock()
_calls = [0]
_raw = [0]
_logged: set = set()


def enabled(layer_id: int) -> bool:
    """Stats dir set and ``layer_id`` in ``SGLANG_TPU_DSA_TOPK_LAYERS`` (a few
    layers by default: the host-side stats take ~0.1 s per call)."""
    if not os.environ.get("SGLANG_TPU_DSA_TOPK_STATS_DIR"):
        return False
    layers = os.environ.get("SGLANG_TPU_DSA_TOPK_LAYERS", "0,10,20,30,40,50,60,70")
    return layer_id in {int(x) for x in layers.split(",") if x}


def _rank() -> int:
    import torch.distributed as dist

    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _consume(layer_id: int, topk: torch.Tensor) -> None:
    out_dir = os.environ["SGLANG_TPU_DSA_TOPK_STATS_DIR"]
    if _rank() != 0 or not os.path.exists(os.path.join(out_dir, "ENABLE")):
        return
    if _calls[0] >= int(os.environ.get("SGLANG_TPU_DSA_TOPK_MAX_CALLS", "400")):
        return
    rows = topk.cpu().numpy()
    live = (rows >= 0).any(axis=1)
    rows = rows[live]  # padded token rows are all -1
    blocks = []
    for b in range(0, rows.shape[0], BLOCK):
        blk = rows[b : b + BLOCK]
        valid = blk[blk >= 0]
        overlap = []
        for i in range(1, blk.shape[0]):
            a, p = blk[i][blk[i] >= 0], blk[i - 1][blk[i - 1] >= 0]
            if a.size:
                overlap.append(np.intersect1d(a, p, assume_unique=True).size / a.size)
        blocks.append(
            dict(
                queries=int(blk.shape[0]),
                valid=int(valid.size),
                distinct=int(np.unique(valid).size),
                max_pos=int(valid.max()) if valid.size else -1,
                adj_overlap=round(float(np.mean(overlap)), 4) if overlap else None,
            )
        )
    # Distinct rows per group of G consecutive queries (deep-context calls only:
    # a group's set grows with context, and shallow queries say little).
    groups = {}
    if rows.size and int(rows.max()) >= 32768:
        for g in GROUPS:
            d = [np.unique(rows[b : b + g][rows[b : b + g] >= 0]).size
                 for b in range(0, rows.shape[0] - g + 1, g)]
            groups[str(g)] = round(float(np.mean(d)), 1) if d else None
    with _lock:
        n = _calls[0]
        _calls[0] += 1
        rec = dict(call=n, layer=layer_id, tokens=int(rows.shape[0]), blocks=blocks,
                   groups=groups)
        with open(os.path.join(out_dir, "stats-rank0.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        # Raw rows only for deep-context chunks (a query near the start of the
        # sequence attends to its whole context, which says nothing).
        deep = rows.size and int(rows.max()) >= 32768
        if deep and _raw[0] < int(os.environ.get("SGLANG_TPU_DSA_TOPK_RAW_LIMIT", "8")):
            _raw[0] += 1
            np.save(os.path.join(out_dir, f"raw-{n:04d}-layer{layer_id}.npy"), rows)


def _layers() -> set[int]:
    # Stats run on the host after every recorded step; a few layers spread
    # over the depth keep serving fast. Static at trace time.
    spec = os.environ.get("SGLANG_TPU_DSA_TOPK_LAYERS", "0,3,10,20,40,60,77")
    return {int(x) for x in spec.split(",") if x}


def record(layer_id: int, topk: torch.Tensor) -> None:
    """Register ``topk`` for the host callback (direct trace), or consume it
    now (eager). Inside any other compiler trace it is skipped."""
    if layer_id not in _layers():
        return
    from sglang_tpu.compilation.direct_compile import insert_callback

    cb = lambda t, _l=layer_id: _consume(_l, t)  # noqa: E731
    registered = insert_callback(topk, cb)
    if _rank() == 0 and layer_id not in _logged:
        _logged.add(layer_id)
        import logging

        logging.getLogger(__name__).info(
            "[dsa_topk_stats] layer %d: callback registered=%s compiling=%s dir=%s",
            layer_id, registered, torch.compiler.is_compiling(),
            os.environ.get("SGLANG_TPU_DSA_TOPK_STATS_DIR"))
    if registered:
        return
    if not torch.compiler.is_compiling():
        _consume(layer_id, topk)
