#!/usr/bin/env bash
set -euo pipefail

PLATFORM_REPO="${PLATFORM_REPO:-edu-llm/platform}"
WORKFLOW="${WORKFLOW:-block-run.yml}"
REGION="${REGION:-us-east-2}"
REPOSITORY="${REPOSITORY:-edu-llm/nested-learning}"
BRANCH="${BRANCH:-edullm/nested-learning-block-12h}"
NODE="${NODE:-}"
RUN_NAME="${RUN_NAME:-nl-smoke-$(date -u +%Y%m%d-%H%M)}"
WANDB_PROJECT="${WANDB_PROJECT:-nested-learning-block-smoke}"
EXECUTE="${EXECUTE:-0}"

SMOKE_OUT_DIR="${SMOKE_OUT_DIR:-configs/generated_block_smoke}"
SMOKE_CONFIG="${SMOKE_CONFIG:-${SMOKE_OUT_DIR}/cms_seed17.json}"
MAX_STEPS="${MAX_STEPS:-2}"
EVAL_EVERY="${EVAL_EVERY:-1}"
EVAL_SIZE="${EVAL_SIZE:-2}"
MAX_LENGTH="${MAX_LENGTH:-512}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
RANK="${RANK:-16}"

if [[ -z "${NODE}" ]]; then
  echo "Set NODE to a free block node first, e.g. NODE=5." >&2
  exit 2
fi

if [[ "${EXECUTE}" == "1" ]] && ! command -v gh >/dev/null 2>&1; then
  echo "EXECUTE=1 needs the GitHub CLI: gh" >&2
  exit 2
fi

command=$(
  printf "bash -lc 'python3 scripts/make_block_12h_configs.py"
  printf " --out-dir %q" "${SMOKE_OUT_DIR}"
  printf " --max-steps %q" "${MAX_STEPS}"
  printf " --eval-every %q" "${EVAL_EVERY}"
  printf " --eval-size %q" "${EVAL_SIZE}"
  printf " --max-length %q" "${MAX_LENGTH}"
  printf " --gradient-accumulation-steps %q" "${GRADIENT_ACCUMULATION_STEPS}"
  printf " --rank %q" "${RANK}"
  printf " && bash scripts/run_block_12h_variant.sh %q'" "${SMOKE_CONFIG}"
)

args=(
  workflow run "${WORKFLOW}"
  --repo "${PLATFORM_REPO}"
  --ref main
  -f "node=${NODE}"
  -f "branch=${BRANCH}"
  -f "run_name=${RUN_NAME}"
  -f "repository=${REPOSITORY}"
  -f "command=${command}"
  -f "processes=1"
  -f "wandb_project=${WANDB_PROJECT}"
  -f "region=${REGION}"
  -f "take_the_node_anyway=false"
)

if [[ -n "${RESERVATION_ID:-}" ]]; then
  args+=(-f "reservation_id=${RESERVATION_ID}")
fi

echo "platform_repo=${PLATFORM_REPO}"
echo "repository=${REPOSITORY}"
echo "branch=${BRANCH}"
echo "node=${NODE}"
echo "run_name=${RUN_NAME}"
echo "execute=${EXECUTE}"
echo "command=${command}"

if [[ "${EXECUTE}" == "1" ]]; then
  gh "${args[@]}"
else
  printf "gh"
  printf " %q" "${args[@]}"
  printf "\n"
  echo "Dry output only. Re-run with EXECUTE=1 after the node is confirmed free."
fi
