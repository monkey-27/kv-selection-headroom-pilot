#!/usr/bin/env bash
#SBATCH --job-name=pdvt-audit
#SBATCH --account=qdp-alpha
#SBATCH --partition=gpuq,gpu_preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-pdvt-audit-%j.out
set -euo pipefail
: "${PDVT_ROOT:?}" "${PDVT_PYTHON:?}" "${PDVT_REPO:?}"
cd "$PDVT_REPO"; export PYTHONPATH="$PDVT_REPO${PYTHONPATH:+:$PYTHONPATH}"
tmp="$PDVT_ROOT/configs/hardware.json.tmp"
"$PDVT_PYTHON" -m pd_vs_tri.hardware > "$tmp"
mv "$tmp" "$PDVT_ROOT/configs/hardware.json"
cat "$PDVT_ROOT/configs/hardware.json"

