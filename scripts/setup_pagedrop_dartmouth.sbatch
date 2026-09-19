#!/usr/bin/env bash
#SBATCH --job-name=pagedrop-setup
#SBATCH --partition=standard
#SBATCH --account=qdp-alpha
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-pagedrop-setup-%j.out

set -euo pipefail
ROOT=/dartfs-hpc/scratch/f0082p5/pagedrop16
cd "$ROOT"
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir --upgrade pip
.venv/bin/python -m pip install --no-cache-dir \
  'torch==2.9.1' 'transformers==5.0.0' accelerate numpy sympy latex2sympy2 \
  regex word2number sentencepiece protobuf 'flashinfer-python==0.6.18' ninja
export HF_HOME="$ROOT/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
.venv/bin/python - <<'PY'
import torch, transformers, flashinfer
from huggingface_hub import snapshot_download
print('environment', torch.__version__, transformers.__version__, flashinfer.__version__, flush=True)
path=snapshot_download('Qwen/Qwen3-4B',revision='1cfa9a7208912126459214e8b04321603b3df60c')
print('model_snapshot',path,flush=True)
PY
