# SPDX-License-Identifier: Apache-2.0
"""The device the copied source DSA kernels are compiled for.

Not part of the source. The source's dispatch reads ``jax.default_backend()``
and ``jax.devices()[0].device_kind``. A torch_tpu op exports its JAX function
with ``JAX_PLATFORMS=cpu`` for a TPU target, where both name the CPU, so every
Pallas / SparseCore branch would fall back to XLA without an error.

``export_target(device_kind)`` names the real target for the duration of one
export; the torch op passes the kind torch_tpu reports for the attached chip.
Outside it ``targets_tpu`` / ``target_device_kind`` return exactly what the
source reads, so plain JAX calls dispatch as in the source. Only the patched
dispatch points (PATCHES.md) read these; nothing global is replaced.
"""

import contextvars
from contextlib import contextmanager

import jax

_EXPORT_DEVICE_KIND: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "source_dsa_export_device_kind", default=None
)


def _clear_dispatching_traces():
    """jax.jit does not key its cache on this context, so drop traces at each boundary."""
    from sglang_tpu.attention.pallas.source_dsa.kernels.dsa import indexer, paged_score

    for fn in (
        indexer.compute_scores_and_select_topk_indices,
        paged_score.paged_extend_score_uniform_packed_pallas,
    ):
        fn.clear_cache()


@contextmanager
def export_target(device_kind: str):
    """Dispatch for a TPU of ``device_kind`` (JAX spelling, e.g. "TPU7x") while exporting."""
    if "TPU" not in device_kind:
        raise ValueError(f"export target must be a TPU, got {device_kind!r}")
    token = _EXPORT_DEVICE_KIND.set(device_kind)
    _clear_dispatching_traces()
    try:
        yield
    finally:
        _EXPORT_DEVICE_KIND.reset(token)
        _clear_dispatching_traces()


def targets_tpu() -> bool:
    """The source's ``jax.default_backend() == "tpu"``, or True while exporting for a TPU."""
    return _EXPORT_DEVICE_KIND.get() is not None or jax.default_backend() == "tpu"


def target_device_kind() -> str:
    """The source's ``jax.devices()[0].device_kind``, or the export target's kind."""
    kind = _EXPORT_DEVICE_KIND.get()
    return kind if kind is not None else jax.devices()[0].device_kind
