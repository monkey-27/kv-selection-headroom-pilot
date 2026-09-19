#!/usr/bin/env bash
#SBATCH --job-name=pagedrop16
#SBATCH --partition=l40s_nova
#SBATCH --account=qdp-alpha
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-pagedrop-%j.out
#SBATCH --requeue

set -euo pipefail

: "${PD_PYTHON:?Set PD_PYTHON to the validated Python environment}"
: "${PD_RUN_ROOT:?Set PD_RUN_ROOT to a persistent scratch output directory}"
: "${PD_MATH_DATA:?Set PD_MATH_DATA to the frozen MATH500 JSONL}"
: "${PD_SYNTH_DATA:?Set PD_SYNTH_DATA to the frozen passcode fixtures}"
: "${PD_LEDGER_HOURS:?Set the approved GPU-hour ledger limit}"
: "${PD_SHARD_ID:?Set the disjoint shard index}"
: "${PD_SHARDS:?Set the common shard count}"
: "${PD_CAP_OVERRIDE_APPROVED:?The user must have approved lifting the original cap}"

test -s "$PD_MATH_DATA"
test -s "$PD_SYNTH_DATA"
mkdir -p "$PD_RUN_ROOT"
cd /dartfs-hpc/scratch/f0082p5/pagedrop16
export PYTHONPATH="$PWD:$PWD/kvcompress/harness${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HF_HOME="${HF_HOME:-$PWD/hf-cache}"
export HF_HUB_CACHE="$HF_HOME/hub"
unset KVH_MODAL_VOLUME_NAME
"$PD_PYTHON" -c 'import torch, flashinfer, transformers; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_memory, flush=True)'
exec "$PD_PYTHON" -m pagedrop_dartmouth
