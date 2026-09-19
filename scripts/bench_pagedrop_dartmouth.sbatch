#!/usr/bin/env bash
#SBATCH --job-name=pagedrop-bench
#SBATCH --partition=l40s_nova
#SBATCH --account=qdp-alpha
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-pagedrop-bench-%j.out

set -euo pipefail
ROOT=/dartfs-hpc/scratch/f0082p5/pagedrop16
cd "$ROOT"
export PD_PILOT_ROOT="$ROOT"
export HF_HOME="$ROOT/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export PYTHONPATH="$ROOT:$ROOT/kvcompress/harness"
"$ROOT/.venv/bin/python" -m pagedrop_dartmouth_bench
