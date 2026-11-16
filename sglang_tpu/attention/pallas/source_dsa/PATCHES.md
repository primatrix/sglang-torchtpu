# Changes to the copied Source files

Source: primatrix-glm/sglang-jax-private at
`c948bdda95208e68a474c8088937347d08fe8937`, `python/sgl_jax/srt/`.
`MANIFEST.json` lists each file's Source sha256. `/tmp`-free check: drop the
4-line header, undo the patches below and the import prefix, and the sha256
must match (the verify script in the M3-S commit does exactly that).

## 1. Import prefix (every file)

`sgl_jax.srt.` -> `sglang_tpu.attention.pallas.source_dsa.`. No other text.

## 2. Target-device dispatch (5 files, 8 lines)

Why: a torch_tpu op exports its JAX function with `JAX_PLATFORMS=cpu` for a TPU
target. There `jax.default_backend()` is `"cpu"` and `jax.devices()[0]` is a CPU
device, so every Pallas / SparseCore branch below would silently take the XLA
fallback.

How: each check reads `target_device.py` instead. Outside an export the helpers
return exactly what the Source read (`jax.default_backend() == "tpu"`,
`jax.devices()[0].device_kind`), so plain JAX calls dispatch as in the Source.
Inside `export_target(kind)` they name the TPU whose kind torch_tpu reports for
the attached chip (`pallas_export.get_pallas_tpu_device_kind`). Nothing global
is replaced, no device kind is hard-coded, and every shape, dtype, page-size and
v7x capability condition next to these checks is unchanged.

| File | Line(s) | Source | Now |
|---|---|---|---|
| `kernels/dsa/indexer.py` | 295, 422, 653 | `jax.default_backend() == "tpu"` | `targets_tpu()` |
| `kernels/dsa/physical_mapping.py` | 90 | `jax.default_backend() == "tpu"` | `targets_tpu()` |
| `kernels/dsa/paged_score.py` | 899 | `jax.default_backend() == "tpu"` | `targets_tpu()` |
| `kernels/dsa/paged_score.py` | 900 | `jax.devices()[0].device_kind in (...)` | `target_device_kind() in (...)` |
| `kernels/dsa/topk.py` | 66 | `jax.devices()[0].device_kind in (...)` | `target_device_kind() in (...)` |
| `kernels/radix_topk/tuned_configs.py` | 66 | `kind = jax.devices()[0].device_kind` | `kind = target_device_kind()` |

Each patched file also gains one import line from `target_device`.

`jax.jit` does not key its cache on the export context, so `export_target`
clears the traces of the two jitted entry points that dispatch in their body
(`compute_scores_and_select_topk_indices`,
`paged_extend_score_uniform_packed_pallas`) on entry and exit.

Remove when: torch_tpu exports with the real TPU backend visible to JAX.

## Kept as in the Source

`@dataclass` (repo rule prefers `msgspec.Struct`) and the `absl` logging import
in radix v1: changing them would break the byte comparison with the Source for
no behavioural gain.
