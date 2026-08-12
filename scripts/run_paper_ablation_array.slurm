#!/usr/bin/env bash
#SBATCH --job-name=olmoe-nl-paper
#SBATCH --array=0-35
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm-%x-%A-%a.out

set -euo pipefail

MANIFEST="${MANIFEST:-configs/generated_paper_ablation/manifest.txt}"
CONFIG="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST}")"

if [[ -z "${CONFIG}" ]]; then
  echo "No config for array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

export CONFIG
echo "Running ${CONFIG}"
bash scripts/run_olmoe_cms_cluster.slurm
