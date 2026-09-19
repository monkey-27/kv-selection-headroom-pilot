#!/usr/bin/env bash
set -euo pipefail
: "${PDVT_ROOT:?}" "${PDVT_REPO:?}" "${PDVT_PYTHON:?}"
while [[ ! -f "$PDVT_ROOT/STOP_PROGRESS" ]]; do
  cd "$PDVT_REPO"
  "$PDVT_PYTHON" -m pd_vs_tri.progress --root "$PDVT_ROOT" --config "$PDVT_ROOT/configs/pd_vs_tri_headline.resolved.json" --hardware "$PDVT_ROOT/configs/hardware.json"
  sleep 600
done
