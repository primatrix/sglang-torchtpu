# SPDX-License-Identifier: Apache-2.0
# Verbatim copy of primatrix-glm/sglang-jax-private (Apache-2.0) at c948bdda95208e68a474c8088937347d08fe8937:
#   python/sgl_jax/srt/kernels/dsa/partial_attention.py
# Only change: import prefix 'sgl_jax.srt.' -> 'sglang_tpu.attention.pallas.source_dsa.'. See source_dsa/MANIFEST.json.
"""FP32 partial-softmax contract shared by the DSA kernels and DCP merge."""

import jax.numpy as jnp


def normalize_attention_output(accumulator, maximum, denominator, *, return_lse=False):
    """Normalize online-softmax state before any output cast.

    A partial kernel appends a 128-lane broadcast of its natural-log LSE to
    the output tile. This keeps the existing aligned, double-buffered output
    DMA; the public wrapper strips the padding and returns a separate LSE.
    Empty partitions have exactly zero output and negative-infinite LSE.
    """
    safe_denominator = jnp.maximum(denominator, jnp.float32(1e-30))
    output = jnp.where((denominator > 0)[..., None], accumulator / safe_denominator[..., None], 0.0)
    if not return_lse:
        return output
    lse = jnp.where(denominator > 0, maximum + jnp.log(safe_denominator), -jnp.inf)
    return jnp.concatenate((output, jnp.broadcast_to(lse[..., None], (*lse.shape, 128))), -1)


def merge_attention_partials(output0, lse0, output1, lse1):
    """Merge two normalized FP32 outputs, including empty partitions.

    ``lse`` is the natural logarithm of the corresponding softmax denominator,
    with the same scale on both ranks. Keep the result FP32 until the model
    explicitly casts it for its value/output projection.
    """
    if output0.shape != output1.shape or lse0.shape != lse1.shape:
        raise ValueError("DCP partial outputs and LSE must have matching shapes")
    if output0.shape[:-1] != lse0.shape:
        raise ValueError("DCP LSE must have shape output.shape[:-1]")
    if any(x.dtype != jnp.float32 for x in (output0, output1, lse0, lse1)):
        raise TypeError("DCP partial outputs and LSE must remain float32 until merge")
    maximum = jnp.maximum(lse0, lse1)
    maximum = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
    a0, a1 = jnp.exp(lse0 - maximum), jnp.exp(lse1 - maximum)
    denominator = jnp.maximum(a0 + a1, jnp.float32(1e-30))
    return (a0[..., None] * output0 + a1[..., None] * output1) / denominator[..., None]
