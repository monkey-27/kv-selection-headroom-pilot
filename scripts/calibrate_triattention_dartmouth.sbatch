#!/usr/bin/env bash
#SBATCH --job-name=tri-cal
#SBATCH --account=qdp-alpha
#SBATCH --partition=gpuq,gpu_preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-tri-cal-%A_%a.out
set -euo pipefail
: "${PDVT_ROOT:?}" "${PDVT_CALIBRATION_TEXT:?}" "${PDVT_MODEL_PATHS_JSON:?}"
models=("Qwen/Qwen3-4B" "Qwen/Qwen3-14B" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "microsoft/Phi-4-reasoning-plus")
model="${models[$SLURM_ARRAY_TASK_ID]}"
safe="${model//\//__}"; out="$PDVT_ROOT/calibration/$safe.pt"; mkdir -p "$(dirname "$out")"
model_path=$("$PDVT_ROOT/envs/triattention/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$PDVT_MODEL_PATHS_JSON" "$model")
"$PDVT_ROOT/envs/triattention/bin/python" "$PDVT_ROOT/checkouts/triattention/scripts/calibrate.py" --model "$model_path" --input "$PDVT_CALIBRATION_TEXT" --output "$out.tmp" --max-length 32768 --device cuda
mv "$out.tmp" "$out"; sha256sum "$out" > "$out.sha256"
"$PDVT_ROOT/envs/triattention/bin/python" - "$out" "$model" <<'PY'
import sys,torch
p,m=sys.argv[1:]; x=torch.load(p,map_location='cpu',weights_only=False)
assert isinstance(x,dict) and 'metadata' in x and 'stats' in x
text=str(x['metadata']); assert m.split('/')[-1].lower().replace('-','') in text.lower().replace('-',''), (m,x['metadata'])
print(x['metadata'])
PY
