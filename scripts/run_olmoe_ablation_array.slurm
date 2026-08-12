#!/usr/bin/env bash
#SBATCH --job-name=olmoe-nl-ablate
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm-%x-%A-%a.out

set -euo pipefail

CONFIGS=(
  "configs/olmoe_static_adapter_cluster.json"
  "configs/olmoe_single_memory_cluster.json"
  "configs/olmoe_cms_cluster.json"
)

export CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Running ${CONFIG}"

bash scripts/run_olmoe_cms_cluster.slurm
