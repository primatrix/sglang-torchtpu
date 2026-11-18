#!/usr/bin/env bash
# Isolated Falcon reproduction of the historical layout benchmark, then fair A/B.
set -euo pipefail
ulimit -c 0

REPO=$(git rev-parse --show-toplevel)
cd "$REPO"
ROOT="${ARTIFACT_LOCAL_DIR:-/tmp/dsa-fair-artifact}"
TASK_TMP=/dev/shm/dsa-fair-share-ab
TASK_VENV="$TASK_TMP/venv"
mkdir -p "$ROOT/profiling" "$ROOT/rank-0/benchmark" \
  "$ROOT/rank-0/profiling/legacy" "$TASK_TMP/tmp" "$TASK_TMP/uv-cache"
export TMPDIR="$TASK_TMP/tmp" UV_CACHE_DIR="$TASK_TMP/uv-cache" UV_NO_CACHE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/python:$REPO"
INSTALL_LOG="$ROOT/profiling/install.log"
git rev-parse HEAD | tee "$ROOT/source-revision.txt"
df -h /tmp /workspace /dev/shm 2>/dev/null | tee "$ROOT/profiling/disk-before.txt"

# The same wheel and TPU package pins as exp-2sgevbvpt5. Only the read-only
# Falcon mount is accepted; do not fall back to provider APIs or another wheel.
WHEEL_NAME=torch_tpu-0.2.1.dev20261028201649-cp312-cp312-manylinux_2_31_x86_64.whl
WHEEL_MOUNT="/wheels/torch-tpu/$WHEEL_NAME"
WHEEL_SHA=6e12816e92f9a5d9649358541f332f06583ccdfee5b6a74434dbf2141757fd82
if [[ ! -f "$WHEEL_MOUNT" ]]; then
  echo "Required Falcon wheel mount is missing: $WHEEL_MOUNT" >&2
  exit 2
fi
printf '%s  %s\n' "$WHEEL_SHA" "$WHEEL_MOUNT" | sha256sum --check | tee "$ROOT/profiling/wheel-sha256.txt"

python3 -m pip install -q uv > "$INSTALL_LOG" 2>&1
uv venv --python 3.12 "$TASK_VENV" >> "$INSTALL_LOG" 2>&1
cp python/pyproject.toml "$TASK_TMP/pyproject-original.toml"
restore_project_manifest() {
  cp "$TASK_TMP/pyproject-original.toml" "$REPO/python/pyproject.toml"
}
trap restore_project_manifest EXIT
cp python/pyproject_tpu.toml python/pyproject.toml
if ! VIRTUAL_ENV="$TASK_VENV" SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev0 uv pip install \
  --index-strategy unsafe-best-match \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "$WHEEL_MOUNT" -e ./python -e . msgspec absl-py >> "$INSTALL_LOG" 2>&1; then
  tail -n 60 "$INSTALL_LOG" >&2
  exit 2
fi
restore_project_manifest
trap - EXIT
PY="$TASK_VENV/bin/python"
uv pip freeze --python "$PY" > "$ROOT/profiling/python-packages.txt"
"$PY" - <<'PY' | tee "$ROOT/profiling/runtime-pins.json"
import importlib.metadata as md
import json

expected = {
    "torch-tpu": "0.2.1.dev20261028201649",
    "torch": "2.11.0",
    "jax": "0.11.2",
    "jaxlib": "0.11.2",
    "libtpu": "0.0.50",
}
actual = {name: md.version(name) for name in expected}
print(json.dumps(actual, sort_keys=True))
for name, version in expected.items():
    # The CPU torch wheel may add the local version suffix +cpu.
    assert actual[name].split("+", 1)[0] == version, (name, actual[name], version)
PY
"$PY" - <<'PY' > "$ROOT/profiling/env.txt"
import os
import platform
import sys

print("python", sys.version)
print("platform", platform.platform())
for name in (
    "JAX_PLATFORMS", "XLA_FLAGS", "LIBTPU_INIT_ARGS", "TPU_LIBRARY_PATH",
    "PYTHONPATH", "TMPDIR", "UV_NO_CACHE",
):
    print(f"{name}={os.environ.get(name, '')}")
PY
# Preserve the historical compiler environment for the first reproduction.
# Full traces and the common benchmark's compiler text are artifact outputs;
# no additional performance-affecting compiler flags are introduced here.
df -h /tmp /dev/shm | tee "$ROOT/profiling/disk-after-install.txt"

if [[ "${1:-}" == gather-phase ]]; then
  "$PY" benchmark/tpu/dsa_k_prologue/bench_fair_shared_ab.py --gather-phase \
    --raw-dir /old/fj8 --out-dir "$ROOT/rank-0" \
    --files synthetic,raw-0096-layer0,raw-0097-layer10,raw-0099-layer40,masked \
    --iters 30 --repeats 10 --profile-iters 0 2>&1 | tee "$ROOT/rank-0/benchmark/phase.log"
  exit 0
fi

"$PY" benchmark/tpu/dsa_k_prologue/bench_q_layout_ab.py \
  --xprof-dir "$ROOT/rank-0/profiling/legacy" --iters 10 --repeats 5 \
  2>&1 | tee "$ROOT/rank-0/benchmark/legacy_q_layout.log"
"$PY" - "$ROOT/rank-0/benchmark/legacy_q_layout.log" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    rows = [json.loads(line[4:]) for line in stream if line.startswith("QAB ")]
assert {row["variant"] for row in rows} == {"q_default", "q_rowmajor", "q_split"}, rows
assert len(rows) == 3, rows
for row in rows:
    assert "fail" not in row and row.get("us", 0) > 0, row
    assert row.get("exact_vs_default") is True, row
print("FAIR_LEGACY_REPRODUCTION_OK", flush=True)
PY

"$PY" benchmark/tpu/dsa_k_prologue/bench_fair_shared_ab.py \
  --raw-dir /old/fj8 --out-dir "$ROOT/rank-0" \
  --groups 2,4,8,16 --iters 20 --repeats 7 --profile-iters 10 \
  2>&1 | tee "$ROOT/rank-0/benchmark/fair_shared_ab.log"
