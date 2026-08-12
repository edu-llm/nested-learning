#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install -r requirements-olmoe.txt
accelerate launch --num_processes 1 train_olmoe_nested.py \
  --config configs/olmoe_cms_smoke.json
