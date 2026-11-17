#!/usr/bin/env bash
set -euo pipefail
cd /workspace/sglang-torchtpu
# Test only: every 2 min, copy the tails of the server / client / runner logs and
# the top-k stats line count to stdout, so `falcon exp logs` shows where a run is
# (the artifact bucket is not readable from outside).
(
  while sleep 120; do
    echo "==== PROGRESS $(date +%T) rank=${FALCON_RANK:-?}"
    find "$ARTIFACT_LOCAL_DIR" \( -name 'server.log' -o -name 'client-attempt-*.log' \
      -o -name 'rank-*-application.log' \) 2>/dev/null | while read -r f; do
      echo "== $f"; tail -n 6 "$f" | cut -c1-300
    done
    find "$ARTIFACT_LOCAL_DIR" -name 'stats-rank0.jsonl' -exec wc -l {} \; 2>/dev/null
    find "$ARTIFACT_LOCAL_DIR" -name 'Traceback*' 2>/dev/null
    grep -rh -m3 -A3 'Traceback\|Error' "$ARTIFACT_LOCAL_DIR" --include='*.log' 2>/dev/null | tail -n 12 | cut -c1-300 || true
  done
) &
exec python3 -u -m benchmark.tpu.glm52_baseline.allocation
