#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-}"
if [[ -z "${CONFIG}" ]]; then
  echo "usage: bash scripts/run_block_12h_variant.sh <config.json>" >&2
  exit 2
fi

if [[ ! -f "${CONFIG}" && "${CONFIG}" == configs/generated_block_12h/* ]]; then
  python3 scripts/make_block_12h_configs.py
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "config does not exist: ${CONFIG}" >&2
  exit 2
fi

mkdir -p results

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PIP_DISABLE_PIP_VERSION_CHECK="${PIP_DISABLE_PIP_VERSION_CHECK:-1}"
export HF_HOME="${HF_HOME:-/scratch/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_PROJECT="${WANDB_PROJECT:-${EDULLM_WANDB_PROJECT:-nested-learning-block}}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${EDULLM_RUN_ID:-$(basename "${CONFIG}" .json)}}"
export WANDB_NAME="${WANDB_NAME:-${WANDB_RUN_ID}}"

if [[ -z "${BLOCK_NUM_PROCESSES:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    BLOCK_NUM_PROCESSES="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    BLOCK_NUM_PROCESSES="8"
  fi
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "block_12h_config=${CONFIG}"
echo "block_num_processes=${BLOCK_NUM_PROCESSES}"
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "edullm_run_id=${EDULLM_RUN_ID:-}"
echo "edullm_output_prefix=${EDULLM_OUTPUT_PREFIX:-}"

python3 -m pip install -r requirements-olmoe.txt

accelerate launch \
  --num_processes "${BLOCK_NUM_PROCESSES}" \
  --mixed_precision bf16 \
  train_olmoe_nested.py \
  --config "${CONFIG}"

echo "finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
