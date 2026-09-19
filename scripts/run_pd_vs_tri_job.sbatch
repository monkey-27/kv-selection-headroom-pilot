#!/usr/bin/env bash
#SBATCH --job-name=pdvt-cell
#SBATCH --account=qdp-alpha
#SBATCH --partition=gpuq,gpu_preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-pdvt-%A_%a.out
#SBATCH --requeue
set -euo pipefail
: "${PDVT_ROOT:?}" "${PDVT_REPO:?}" "${PDVT_PYTHON:?}" "${PDVT_JOB_FILE:?}" "${PDVT_MANIFEST:?}" "${PDVT_HARDWARE:?}"
cd "$PDVT_REPO"; export PYTHONPATH="$PDVT_REPO${PYTHONPATH:+:$PYTHONPATH}" TOKENIZERS_PARALLELISM=false
"$PDVT_PYTHON" -m pd_vs_tri.runner --root "$PDVT_ROOT" --job "$PDVT_JOB_FILE" --manifest "$PDVT_MANIFEST" --hardware "$PDVT_HARDWARE"
"$PDVT_PYTHON" -m pd_vs_tri.progress --root "$PDVT_ROOT" --config "$PDVT_ROOT/configs/pd_vs_tri_headline.resolved.json" --hardware "$PDVT_HARDWARE"

