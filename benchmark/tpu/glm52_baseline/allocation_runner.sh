#!/usr/bin/env bash
set -euo pipefail
cd /workspace/sglang-torchtpu
# Test only: every 2 min, copy the tails of the server / client / runner logs and
# the top-k stats line count to stdout, so `falcon exp logs` shows where a run is
# (the artifact bucket is not readable from outside). The GCS mount returns
# "Stale file handle" at times: the loop must never exit on a read error.
(
  set +euo pipefail
  while sleep 120; do
    echo "==== PROGRESS $(date +%T) rank=${FALCON_RANK:-?}"
    for f in $(find "$ARTIFACT_LOCAL_DIR" \( -name 'server.log' -o -name 'client-attempt-*.log' \
        -o -name 'rank-*-application.log' -o -name 'result*.json' \) 2>/dev/null); do
      echo "== $f"; tail -n 6 "$f" 2>/dev/null | cut -c1-300
    done
    for f in $(find "$ARTIFACT_LOCAL_DIR" -name 'stats-rank0.jsonl' 2>/dev/null); do
      echo "STATS $f $(wc -l < "$f" 2>/dev/null)"
    done
    grep -rh 'topk_stats\|dsa_topk' "$ARTIFACT_LOCAL_DIR" --include='server.log' 2>/dev/null | tail -n 4 | cut -c1-300
    grep -rh -B2 -A6 'Traceback' "$ARTIFACT_LOCAL_DIR" --include='*.log' 2>/dev/null | tail -n 20 | cut -c1-300
  done
) &
# Hard bound: the allocation deadline did not stop the last run before Falcon's 2 h limit.
LIMIT=$(( ${GLM52_ALLOCATION_SECONDS:-6900} + 180 ))
exec timeout --signal=TERM --kill-after=60 "$LIMIT" python3 -u -m benchmark.tpu.glm52_baseline.allocation
