#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$ROOT/configs/kv_headroom.json}"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/kvcompress/harness${PYTHONPATH:+:$PYTHONPATH}"

python3 -m kvheadroom.traces --config "$CONFIG"
python3 -m kvheadroom.exhaustive synthetic --config "$CONFIG"
python3 -m kvheadroom.exhaustive main --config "$CONFIG"
python3 -m kvheadroom.analyze --config "$CONFIG"
python3 -m kvheadroom.free_generation --config "$CONFIG"
python3 -m kvheadroom.exhaustive late --config "$CONFIG"
python3 -m kvheadroom.analyze --config "$CONFIG"
