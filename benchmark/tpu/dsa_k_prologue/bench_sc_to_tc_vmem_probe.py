"""Test only: can a SparseCore kernel write into TensorCore VMEM via mpmd_map?

Source reading (jax 0.11.2 / 0.12.0.1 pallas_call_registration.py) says
``_mpmd_map_tpu_lowering`` raises "mpmd_map does not support TC kernels yet."
when a TC mesh is mixed with SC meshes. This records what actually happens on
TPU. Cases:

* ``sc_only``: control. SC vector subcores copy rows HBM -> HBM through their
  own VMEM via mpmd_map (the supported shape), so the harness itself is valid.
* ``tc_sc``: TC mesh + SC vector-subcore mesh in one mpmd_map. SC gathers rows
  into a TC-VMEM scratch (``CoreMemorySpace(VMEM, tc_mesh)``); TC copies the
  scratch to the output. Exact check against numpy if it runs.
* ``sc_tc_vmem_ref``: a plain SC ``pl.kernel`` whose scratch is TC VMEM, to see
  whether the memory space is accepted outside mpmd_map.

Each stage (trace / lower / compile / run) is reported separately. Prints
``SCTC`` JSON lines.
"""

import json
import os
import traceback

os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.experimental.pallas import tpu_sc as plsc  # noqa: E402

ROWS = 64  # rows copied (one 8-row tile per subcore x 8 subcores)
W = 128  # int32 words per row (512 B NoPE)


def report(case, stage, **kw):
    print("SCTC " + json.dumps(dict(case=case, stage=stage, **kw)), flush=True)


def run_stages(case, fn, args, ref=None):
    try:
        jf = jax.jit(fn)
        lowered = jf.lower(*args)
        report(case, "lower", ok=True)
    except Exception as e:  # report, do not hide
        report(case, "lower", ok=False, error=repr(e)[:1500],
               where=traceback.format_exc().splitlines()[-3:])
        return
    try:
        compiled = lowered.compile()
        report(case, "compile", ok=True)
    except Exception as e:
        report(case, "compile", ok=False, error=repr(e)[:1500])
        return
    try:
        out = np.asarray(jax.block_until_ready(compiled(*args)))
        report(case, "run", ok=True,
               exact=None if ref is None else bool(np.array_equal(out, ref)))
    except Exception as e:
        report(case, "run", ok=False, error=repr(e)[:1500])


def main():
    from jax._src.pallas import mpmd  # not exported publicly in 0.11.2
    from jax._src.pallas import core as pallas_core
    from jax._src.pallas.mosaic import core as tpu_core

    sc = pltpu.get_tpu_info().sparse_core
    report("info", "hw", sc_cores=sc.num_cores, sc_subcores=sc.num_subcores,
           devices=str(jax.devices()[:1]))
    rng = np.random.default_rng(0)
    k_np = rng.integers(0, 1000, (1024, W), dtype=np.int32)
    idx_np = rng.choice(1024, ROWS, replace=False).astype(np.int32)
    ref = k_np[idx_np]
    k, idx = jnp.asarray(k_np), jnp.asarray(idx_np)

    vmesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore",
                                   num_cores=1, num_subcores=8)
    tmesh = tpu_core.create_tensorcore_mesh("tc", num_cores=1)

    # --- control: SC-only mpmd_map, HBM -> own VMEM (indexed) -> HBM.
    def sc_body(k_hbm, idx_hbm, o_hbm, idx_v, rows_v, sem):
        s = lax.axis_index("subcore")
        pltpu.sync_copy(idx_hbm.at[pl.ds(s * 8, 8)], idx_v)
        pltpu.async_copy(k_hbm.at[idx_v], rows_v, sem).wait()
        pltpu.sync_copy(rows_v, o_hbm.at[pl.ds(s * 8, 8)])

    def sc_only(k, idx):
        return mpmd.mpmd_map(
            [(vmesh, sc_body)],
            out_types=jax.ShapeDtypeStruct((ROWS, W), jnp.int32),
            scratch_types=[pltpu.VMEM((8,), jnp.int32), pltpu.VMEM((8, W), jnp.int32),
                           pltpu.SemaphoreType.DMA],
            name="sctc_sc_only",
        )(k, idx)

    run_stages("sc_only", sc_only, (k, idx), ref)

    # --- TC + SC in one mpmd_map. With >1 mesh every scratch must name its
    # core (CoreMemorySpace / SemaphoreType @ mesh); the first run left the SC
    # scratch untagged and stopped there.
    tc_vmem = pallas_core.CoreMemorySpace(tpu_core.MemorySpace.VMEM, tmesh)
    sc_vmem = pallas_core.CoreMemorySpace(tpu_core.MemorySpace.VMEM, vmesh)

    def tc_sc_scratch():
        return [tc_vmem((ROWS, W), jnp.int32), sc_vmem((8,), jnp.int32),
                sc_vmem((8, W), jnp.int32), pltpu.SemaphoreType.DMA @ vmesh]

    # (a) indexed gather straight into TC VMEM (expected: the DMA rule refuses).
    def tc_sc(k, idx):
        def sc_part(k_hbm, idx_hbm, o_hbm, tcbuf, idx_v, rows_v, sem):
            s = lax.axis_index("subcore")
            pltpu.sync_copy(idx_hbm.at[pl.ds(s * 8, 8)], idx_v)
            pltpu.async_copy(k_hbm.at[idx_v], tcbuf.at[pl.ds(s * 8, 8)], sem).wait()

        def tc_part(k_hbm, idx_hbm, o_hbm, tcbuf, idx_v, rows_v, sem):
            pltpu.sync_copy(tcbuf, o_hbm)

        return mpmd.mpmd_map(
            [(vmesh, sc_part), (tmesh, tc_part)],
            out_types=jax.ShapeDtypeStruct((ROWS, W), jnp.int32),
            scratch_types=tc_sc_scratch(), name="sctc_tc_sc",
        )(k, idx)

    run_stages("tc_sc", tc_sc, (k, idx), ref)

    # (b) SC gathers into its own VMEM (supported), then a plain copy of the
    # finished tile into TC VMEM; TC copies it out. No cross-core barrier yet,
    # so a mismatch at run may be a race; compile/lower is the main question.
    def tc_sc_plain(k, idx):
        def sc_part(k_hbm, idx_hbm, o_hbm, tcbuf, idx_v, rows_v, sem):
            s = lax.axis_index("subcore")
            pltpu.sync_copy(idx_hbm.at[pl.ds(s * 8, 8)], idx_v)
            pltpu.async_copy(k_hbm.at[idx_v], rows_v, sem).wait()
            pltpu.async_copy(rows_v, tcbuf.at[pl.ds(s * 8, 8)], sem).wait()

        def tc_part(k_hbm, idx_hbm, o_hbm, tcbuf, idx_v, rows_v, sem):
            pltpu.sync_copy(tcbuf, o_hbm)

        return mpmd.mpmd_map(
            [(vmesh, sc_part), (tmesh, tc_part)],
            out_types=jax.ShapeDtypeStruct((ROWS, W), jnp.int32),
            scratch_types=tc_sc_scratch(), name="sctc_tc_sc_plain",
        )(k, idx)

    run_stages("tc_sc_plain", tc_sc_plain, (k, idx), ref)

    # (c) single SC kernel: gather into own VMEM, plain copy into a TC-VMEM
    # scratch, read it back to HBM from SC.
    def sc_plain_tc_ref(k, idx):
        def body(k_hbm, idx_hbm, o_hbm, tcbuf, idx_v, rows_v, sem):
            s = lax.axis_index("subcore")
            pltpu.sync_copy(idx_hbm.at[pl.ds(s * 8, 8)], idx_v)
            pltpu.async_copy(k_hbm.at[idx_v], rows_v, sem).wait()
            pltpu.sync_copy(rows_v, tcbuf.at[pl.ds(s * 8, 8)])
            pltpu.sync_copy(tcbuf.at[pl.ds(s * 8, 8)], o_hbm.at[pl.ds(s * 8, 8)])

        return pl.kernel(
            body,
            out_type=jax.ShapeDtypeStruct((ROWS, W), jnp.int32),
            mesh=vmesh,
            scratch_types=(tc_vmem((ROWS, W), jnp.int32), pltpu.VMEM((8,), jnp.int32),
                           pltpu.VMEM((8, W), jnp.int32), pltpu.SemaphoreType.DMA),
            name="sctc_sc_plain_tc_ref",
        )(k, idx)

    try:
        run_stages("sc_plain_tc_ref", sc_plain_tc_ref, (k, idx), ref)
    except Exception as e:
        report("sc_plain_tc_ref", "build", ok=False, error=repr(e)[:1500])

    print("SCTC_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # build-time failures before any stage
        print("SCTC " + json.dumps(dict(case="main", stage="build", ok=False,
                                       error=repr(e)[:1500],
                                       where=traceback.format_exc().splitlines()[-4:])),
              flush=True)
        print("SCTC_DONE", flush=True)
