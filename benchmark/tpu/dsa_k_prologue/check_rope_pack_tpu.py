"""Test only: why SparseCoreRopePackingTest fails on TPU but passes on CPU.

Replays test_packed_writer_matches_the_host_layout (jnp writer vs
pack_rope_packed) on TPU and on CPU in one process, then lists which source
bytes the TPU write changed. Also checks the bare uint8 -> fp8 -> uint8 bitcast
round trip on TPU, and reruns the writer with NaN-free bytes (no 0x7F / 0xFF).
Prints ``RPC`` lines.
"""

import json
import os

os.environ["JAX_PLATFORMS"] = "tpu,cpu"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from sglang_tpu.attention.pallas.mla_sparse import native_sc_cache_layout  # noqa: E402
from sglang_tpu.attention.pallas.mla_sparse.kv_cache_utils import (  # noqa: E402
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
    update_sparse_mla_kv_cache_jax,
)

PAGE_SIZE, TOTAL_PAGES, ROPE_DIM, LKV_DIM, KV_PACKING = 32, 16, 64, 512, 4
NAN_CODES = (0x7F, 0xFF)


def emit(**kw):
    print("RPC " + json.dumps(kw), flush=True)


def main():
    tpu, cpu = jax.devices("tpu")[0], jax.devices("cpu")[0]
    spec = lambda t, d: SparseMLAKVCacheSpec.create(  # noqa: E731
        t, KVCacheLayout.SPARSECORE, TOTAL_PAGES, PAGE_SIZE, d, KV_PACKING)
    packed, nope = spec(KVCacheType.ROPE, ROPE_DIM), spec(KVCacheType.NOPE, LKV_DIM)
    n = PAGE_SIZE * 2
    raw = np.random.default_rng(3).integers(0, 256, (n, ROPE_DIM), np.uint8)
    bt = np.array([6, 2, 0, 0], np.int32)

    # 1. Bare bitcast round trip per device.
    for name, dev in (("tpu", tpu), ("cpu", cpu)):
        x = jax.device_put(jnp.asarray(raw), dev)
        rt = np.asarray(jax.jit(lambda a: jax.lax.bitcast_convert_type(
            jax.lax.bitcast_convert_type(a, jnp.float8_e4m3fn), jnp.uint8))(x))
        bad = rt != raw
        emit(check="bitcast_roundtrip", device=name, changed=int(bad.sum()),
             changed_src_values=sorted(set(raw[bad].tolist()))[:16],
             changed_to=sorted(set(rt[bad].tolist()))[:16])

    def write(src, dev):
        with jax.default_device(dev):
            k_pe = jax.lax.bitcast_convert_type(jnp.asarray(src), jnp.float8_e4m3fn)
            _, rope = update_sparse_mla_kv_cache_jax(
                jnp.zeros(nope.shape, nope.jax_dtype),
                jnp.zeros(packed.shape, packed.jax_dtype),
                jnp.zeros((n, LKV_DIM), jnp.float8_e4m3fn), k_pe,
                jnp.asarray([n], jnp.int32), jnp.asarray(bt),
                jnp.asarray([0, n], jnp.int32), nope_spec=nope, rope_spec=packed)
            return np.asarray(rope)

    # 2. The failing test's writer, random bytes and NaN-free bytes.
    clean = raw.copy()
    for v in NAN_CODES:
        clean[clean == v] = 0x00
    emit(check="nan_codes_in_input", count=int(np.isin(raw, NAN_CODES).sum()),
         total=int(raw.size))
    for label, src in (("random", raw), ("nan_free", clean)):
        exp = native_sc_cache_layout.pack_rope_packed(src.reshape(2, PAGE_SIZE, ROPE_DIM))
        for name, dev in (("tpu", tpu), ("cpu", cpu)):
            got = write(src, dev)
            ok = bool(np.array_equal(got[6], exp[0]) and np.array_equal(got[2], exp[1]))
            r = dict(check="writer_vs_host", input=label, device=name, match=ok)
            if not ok:
                g = np.concatenate([got[6], got[2]]).view(np.uint8).ravel()
                e = np.concatenate([exp[0], exp[1]]).view(np.uint8).ravel()
                bad = g != e
                r.update(bytes_changed=int(bad.sum()),
                         changed_src_values=sorted(set(e[bad].tolist()))[:16],
                         changed_to=sorted(set(g[bad].tolist()))[:16],
                         all_changed_are_nan_codes=bool(np.isin(e[bad], NAN_CODES).all()))
            emit(**r)
    print("RPC_DONE", flush=True)


if __name__ == "__main__":
    main()
