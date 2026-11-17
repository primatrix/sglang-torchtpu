# Ported from tpu-inference (https://github.com/vllm-project/tpu-inference)
#   tpu_inference/kernels/sparse_core/core_map_helper.py @ 4367dcff1 (2026-07-19).

"""Run single-mesh SparseCore kernels through ``core_map``.

``pl.kernel`` lowers through ``mpmd_map``, which is slower than the ``core_map``
path for these single-mesh kernels on TPU. ``kernel`` here is a drop-in
replacement for ``pl.kernel`` that uses ``core_map`` directly, so the SparseCore
kernels keep that lowering without depending on the jax-side ``pl.kernel``
implementation.
"""

from jax._src import api
from jax._src import core as jax_core
from jax._src import lax, tree_util
from jax._src.pallas import core as pl_core


def _empty_out_ref(out_type, dep=None):
    aval = pl_core._convert_out_shape_to_aval(out_type)
    memory_space = (
        None
        if isinstance(aval.memory_space, jax_core.MemorySpace)
        else aval.memory_space
    )
    value = lax.empty(aval.shape, aval.dtype, out_sharding=aval.sharding)
    if dep is not None:
        # Tie this call's output buffer to its own operands. `empty` has no
        # operands, so XLA merges the identical ones of repeated calls in one
        # graph into a single buffer; every call then writes that shared buffer
        # in place and XLA copies it first (16 x ~1 ms per extend for the native
        # SparseCore gather under torch_tpu, exp-qyuatjpneb). The barrier makes
        # each call's buffer a distinct value; it moves no data.
        import jax

        value, _ = jax.lax.optimization_barrier((value, dep))
    return jax_core.new_ref(value, memory_space=memory_space)


def kernel(
    body,
    *,
    out_type,
    mesh,
    scratch_types=(),
    compiler_params=None,
    interpret=False,
    cost_estimate=None,
    debug=False,
    name=None,
    metadata=None,
    distinct_outputs=False,
):
    """Drop-in replacement for ``pl.kernel`` that lowers via ``core_map``.

    ``distinct_outputs`` makes each call's output buffers depend on its
    operands, so repeated calls in one graph never share (and copy) one
    uninitialized buffer (see ``_empty_out_ref``).
    """
    single_output = not isinstance(out_type, (tuple, list))
    out_types = (out_type,) if single_output else out_type

    @api.jit
    def run(*operands):
        arg_refs = tree_util.tree_map(jax_core.new_ref, operands)
        dep = operands if distinct_outputs else None
        out_refs = tree_util.tree_map(lambda t: _empty_out_ref(t, dep), out_types)

        @pl_core.core_map(
            mesh,
            scratch_shapes=scratch_types,
            compiler_params=compiler_params,
            interpret=interpret,
            cost_estimate=cost_estimate,
            debug=debug,
            name=name,
            metadata=metadata,
        )
        def _(*scratch_refs, **scratch_kwrefs):
            return body(*arg_refs, *out_refs, *scratch_refs, **scratch_kwrefs)

        outs = tree_util.tree_map(lambda ref: ref[...], out_refs)
        return outs[0] if single_output else outs

    return run
