"""Test only: dump the LLO of the TensorCore VMEM row-gather kernel.

bench_tc_vmem_gather.py measured ``vld`` at ~1.46 ns/row. This compiles the same
kernel (unroll 1 and 8) with the LLO text dump on, then prints the dumped files
of the kernel to stdout (the artifact bucket is not readable from outside), so
the per-row bundle count and slot use can be read from ``falcon exp logs``.
Lines are prefixed ``LLO <file>|``.
"""

import glob
import os
import sys

DUMP = "/tmp/llo_dump"
os.makedirs(DUMP, exist_ok=True)
os.environ.setdefault("JAX_PLATFORMS", "tpu")
# --xla_jf_* must go to LIBTPU_INIT_ARGS: they abort the XLA_FLAGS parser.
os.environ["LIBTPU_INIT_ARGS"] = (
    os.environ.get("LIBTPU_INIT_ARGS", "")
    + f" --xla_jf_dump_to={DUMP} --xla_jf_dump_llo_text=true"
    " --xla_mosaic_enable_llo_source_annotations=true"
).strip()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import bench_tc_vmem_gather as vg  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    d = 8192
    k = jnp.asarray(rng.integers(0, 1000, (d, vg.W), dtype=np.int32))
    idx = jnp.asarray(np.stack([rng.choice(d, vg.TOPK, replace=False)
                                for _ in range(vg.Q)]).astype(np.int32).reshape(-1))
    for unroll in (1, 8):
        jax.block_until_ready(vg.gather(k, idx, variant="vld", qb=1, unroll=unroll))
    files = sorted(f for f in glob.glob(f"{DUMP}/**/*", recursive=True) if os.path.isfile(f))
    print(f"LLO_FILES {len(files)}", flush=True)
    for f in files:
        print(f"LLO_FILE {os.path.getsize(f)} {f}", flush=True)
    # The kernel's own files: name contains the pallas_call name.
    mine = [f for f in files if "tc_vmem_gather" in os.path.basename(f)
            or "tc_vmem_gather" in open(f, errors="replace").read(200000)]
    for f in mine:
        text = open(f, errors="replace").read().splitlines()
        tag = os.path.basename(f)[-60:]
        print(f"LLO_BEGIN {len(text)} {f}", flush=True)
        for line in text[:3000]:
            print(f"LLO {tag}|{line[:400]}", flush=True)
        print(f"LLO_END {f}", flush=True)
    print("LLO_DONE", flush=True)


if __name__ == "__main__":
    main()
