#!/usr/bin/env bash
# VisionBox nightly self-training entrypoint (invoked by visionbox-train.timer).
#
# Stages a candidate model by default and writes runs/train/<run>/REPORT.json for
# review. Set AUTO_PROMOTE=1 (see visionbox-train.service) to hot-swap automatically
# whenever the base-class regression gate passes.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="$ROOT/venv/bin/python"
TS="$(date +%Y%m%d_%H%M%S)"

# Refresh the COCO-replay regression set (idempotent; needs internet the first time).
# If it can't be built the gate fails CLOSED (stage-only, no auto-promote) — non-fatal,
# but nothing will be promoted until the replay baseline exists.
"$PY" scripts/training/prep_replay.py \
    || echo "[run_overnight] prep_replay unavailable; gate will fail closed (stage only)"

args=(--run-ts "$TS")
[ "${AUTO_PROMOTE:-0}" = "1" ] && args+=(--auto-promote)

exec "$PY" scripts/training/train_overnight.py "${args[@]}"
