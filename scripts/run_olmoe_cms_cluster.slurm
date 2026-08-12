#!/usr/bin/env bash
#SBATCH --job-name=olmoe-cms
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm-%x-%j.out

set -euo pipefail

module load cuda || true
python3 -m pip install -r requirements-olmoe.txt

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

accelerate launch \
  --num_processes "${SLURM_NTASKS:-8}" \
  --mixed_precision bf16 \
  train_olmoe_nested.py \
  --config "${CONFIG:-configs/olmoe_cms_cluster.json}"
