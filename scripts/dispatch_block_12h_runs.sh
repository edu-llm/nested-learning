#!/usr/bin/env bash
set -euo pipefail

PLAN="${PLAN:-configs/generated_block_12h/dispatch_plan.tsv}"
PLATFORM_REPO="${PLATFORM_REPO:-edu-llm/platform}"
WORKFLOW="${WORKFLOW:-block-run.yml}"
REGION="${REGION:-us-east-2}"
WANDB_PROJECT="${WANDB_PROJECT:-nested-learning-block}"
WAVE="${WAVE:-1}"
EXECUTE="${EXECUTE:-0}"

if [[ ! -f "${PLAN}" ]]; then
  python3 scripts/make_block_12h_configs.py
fi

if [[ -z "${REPOSITORY:-}" ]]; then
  echo "Set REPOSITORY to the public GitHub repo containing this experiment, e.g. REPOSITORY=owner/repo" >&2
  exit 2
fi

if [[ -z "${BRANCH:-}" ]]; then
  echo "Set BRANCH to the pushed branch containing this experiment." >&2
  exit 2
fi

if [[ "${EXECUTE}" == "1" ]] && ! command -v gh >/dev/null 2>&1; then
  echo "EXECUTE=1 needs the GitHub CLI: gh" >&2
  exit 2
fi

echo "plan=${PLAN}"
echo "platform_repo=${PLATFORM_REPO}"
echo "repository=${REPOSITORY}"
echo "branch=${BRANCH}"
echo "wave=${WAVE}"
echo "execute=${EXECUTE}"

tail -n +2 "${PLAN}" | while IFS=$'\t' read -r wave node run_name config; do
  if [[ "${wave}" != "${WAVE}" ]]; then
    continue
  fi

  command="bash scripts/run_block_12h_variant.sh ${config}"
  args=(
    workflow run "${WORKFLOW}"
    --repo "${PLATFORM_REPO}"
    --ref main
    -f "node=${node}"
    -f "branch=${BRANCH}"
    -f "run_name=${run_name}"
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

  if [[ "${EXECUTE}" == "1" ]]; then
    gh "${args[@]}"
  else
    printf "gh"
    printf " %q" "${args[@]}"
    printf "\n"
  fi
done

if [[ "${EXECUTE}" != "1" ]]; then
  echo "Dry output only. Re-run with EXECUTE=1 after verifying the nodes are free."
fi
