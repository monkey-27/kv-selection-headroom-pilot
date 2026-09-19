#!/usr/bin/env bash
set -euo pipefail
: "${PDVT_ROOT:?persistent experiment root required}"
: "${PDVT_PYTHON_MODULE:=python/3.10}"
TRI_SHA=325297218a0d85cc83bc9ca1ecfa1a33a178831f
TRT_SHA=8a26dd8f9d6fd09781d7e6f5f1162674f9f8fd05
mkdir -p "$PDVT_ROOT"/{checkouts,envs,manifests,raw,calibration,smoke,results,runs,reports,configs,tmp}
if [[ ! -d "$PDVT_ROOT/checkouts/triattention/.git" ]]; then git clone https://github.com/WeianMao/triattention.git "$PDVT_ROOT/checkouts/triattention"; fi
git -C "$PDVT_ROOT/checkouts/triattention" fetch --tags origin
git -C "$PDVT_ROOT/checkouts/triattention" checkout --detach "$TRI_SHA"
test "$(git -C "$PDVT_ROOT/checkouts/triattention" rev-parse HEAD)" = "$TRI_SHA"
if [[ "${PDVT_ENABLE_TRTLLM:-0}" == 1 ]]; then
  if [[ ! -d "$PDVT_ROOT/checkouts/TensorRT-LLM/.git" ]]; then git clone https://github.com/NVIDIA/TensorRT-LLM.git "$PDVT_ROOT/checkouts/TensorRT-LLM"; fi
  git -C "$PDVT_ROOT/checkouts/TensorRT-LLM" fetch origin "$TRT_SHA"
  git -C "$PDVT_ROOT/checkouts/TensorRT-LLM" checkout --detach "$TRT_SHA"
  test "$(git -C "$PDVT_ROOT/checkouts/TensorRT-LLM" rev-parse HEAD)" = "$TRT_SHA"
fi
python3 -m venv "$PDVT_ROOT/envs/triattention"
"$PDVT_ROOT/envs/triattention/bin/pip" install --upgrade pip wheel
"$PDVT_ROOT/envs/triattention/bin/pip" install -e "$PDVT_ROOT/checkouts/triattention"
"$PDVT_ROOT/envs/triattention/bin/pip" freeze > "$PDVT_ROOT/configs/triattention.pip-freeze.txt"
