# GitHub Block 12-Hour Nested Learning Run

This runbook uses the platform repository's `Block:` GitHub Actions lane to run
the OLMoE Nested Learning diagnostic matrix inside one capacity-block window.
The local `SKILL.md` applies: I prepared files and read workflow definitions,
but did not dispatch GitHub workflows or touch AWS resources.

## What To Run

Use the default `core8` matrix first. It fills eight nodes with one variant per
node:

1. `base_eval`
2. `static_adapter`
3. `single_memory`
4. `cms`
5. `cms_no_dgd`
6. `cms_no_momentum`
7. `cms_no_retention`
8. `cms_no_self_reference`

This is the fastest high-signal test of whether CMS is doing real nested
learning rather than merely adding parameters or one online memory.

The generated configs use:

- base model: `allenai/OLMoE-1B-7B-0125`
- adapter rank: `256`
- CMS chunk sizes: `128`, `512`, `2048`
- max sequence length: `2048`
- optimization steps: `800`
- eval every: `200`
- eval set size: `128`
- gradient accumulation: `8`

Generate the files:

```bash
python3 scripts/make_block_12h_configs.py
```

The generator writes:

```text
configs/generated_block_12h/manifest.txt
configs/generated_block_12h/dispatch_plan.tsv
configs/generated_block_12h/*.json
```

## Repository Requirement

`Block: start a run on a node` clones a public repository and branch onto the
node. This experiment is published at `edu-llm/nested-learning`; use the pushed
branch below for the 12-hour Block matrix.

Use the current prepared branch:

```bash
export BRANCH=edullm/nested-learning-block-12h
```

Use the public GitHub repository:

```bash
export REPOSITORY=edu-llm/nested-learning
```

## Block Workflow Map

The important `Block:` workflows in `edu-llm/platform` are:

- `block-launch-fleet.yml`: starts the eight-node capacity-block fleet. This is
  the costful mutation and should be run once by the block owner.
- `block-status.yml`: reads every node and reports which are free.
- `block-run.yml`: starts one run on one node. This is what the 12-hour matrix
  uses.
- `block-run-distributed.yml`: starts one multi-node run across several nodes.
  Use this only for a single distributed training run, not for the eight-way
  ablation matrix.
- `block-logs.yml`: prints a run log tail into the GitHub job summary.
- `block-drain.yml`: asks nodes to flush and reports what reached S3.
- `block-release.yml`: releases stale node claims after containers have exited.

For this experiment, prefer `block-run.yml` over `block-run-distributed.yml`.
The distributed workflow appends `torchrun` and, by default, OLMo-core MoE mesh
flags. If you do use it for this repository, set:

```text
mesh_flags: false
command: python train_olmoe_nested.py --config configs/generated_block_12h/cms_seed17.json
```

Do not pass `bash scripts/run_block_12h_variant.sh ...` to the distributed
workflow; that workflow expects a Python training entrypoint, not a shell helper.

## Launch Or Reuse The Fleet

If the fleet is already up, skip this section. If it is not up, the block owner
uses:

```text
Block: launch the fleet
```

Typical inputs:

```text
reservation_id: cr-...
instance_type: p5.48xlarge
availability_zone: us-east-2a
instance_count: 8
image_tag: <approved OLMo-core image tag>
region: us-east-2
outputs_bucket: edullm-block-outputs-us-east-2
data_bucket: edullm-data-us-east-2
subnet_id: leave empty unless the tagged subnet is ambiguous
ami_id: leave empty unless overriding the current AWS Deep Learning Base OSS NVIDIA AMI
efa_interfaces: leave empty for the instance-type default
```

The launch workflow intentionally does not use a cluster placement group; the
capacity block is already delivered with the expected locality, and the workflow
verifies that launched instances are drawing from the named capacity reservation.

## Preflight The Fleet

From `edu-llm/platform` Actions, run:

```text
Block: which node is free
```

Inputs:

```text
reservation_id: leave empty unless multiple capacity blocks are live
region: us-east-2
```

Only proceed if nodes `1-8` are free or explicitly assigned to this experiment.
Do not use `take_the_node_anyway`.

## Smoke Test First

Mirror the safe path from the earlier block run: claim one free node, run a tiny
two-step CMS job, read the log, and release the node after the container exits.
This verifies the public branch, dependency install, OLMoE load, adapter
injection, synthetic episodic data path, evaluation path, and block output sync
before occupying the whole fleet.

First print the exact dispatch command:

```bash
NODE=5 \
REPOSITORY=edu-llm/nested-learning \
BRANCH=edullm/nested-learning-block-12h \
bash scripts/dispatch_block_smoke.sh
```

After checking that node is free, dispatch the smoke:

```bash
NODE=5 \
REPOSITORY=edu-llm/nested-learning \
BRANCH=edullm/nested-learning-block-12h \
EXECUTE=1 \
bash scripts/dispatch_block_smoke.sh
```

The smoke command generated on the node is:

```bash
bash -lc 'python3 scripts/make_block_12h_configs.py --out-dir configs/generated_block_smoke --max-steps 2 --eval-every 1 --eval-size 2 --max-length 512 --gradient-accumulation-steps 1 --rank 16 && bash scripts/run_block_12h_variant.sh configs/generated_block_smoke/cms_seed17.json'
```

Use a unique `RUN_NAME` if re-running, for example:

```bash
RUN_NAME=nl-smoke-20260812-0600
```

After the smoke run exits, release only that node:

```text
Block: give a node back
nodes: 5
region: us-east-2
```

## Dispatch The Eight Runs

First print the exact GitHub CLI commands:

```bash
REPOSITORY=edu-llm/nested-learning \
BRANCH=edullm/nested-learning-block-12h \
bash scripts/dispatch_block_12h_runs.sh
```

After checking the output, dispatch:

```bash
REPOSITORY=edu-llm/nested-learning \
BRANCH=edullm/nested-learning-block-12h \
EXECUTE=1 \
bash scripts/dispatch_block_12h_runs.sh
```

Optional inputs:

```bash
export RESERVATION_ID=cr-...
export REGION=us-east-2
export WANDB_PROJECT=nested-learning-block
```

Each node receives a command like:

```bash
bash scripts/run_block_12h_variant.sh configs/generated_block_12h/cms_seed17.json
```

The Block workflow should use:

```text
workflow: block-run.yml
processes: 1
take_the_node_anyway: false
```

`processes=1` is deliberate: the script installs dependencies and then starts
`accelerate launch` itself across the visible GPUs. Do not set `processes=all`
for this helper script.

## Monitor

For status, run:

```text
Block: which node is free
```

For logs, run:

```text
Block: read a run's log
```

Useful inputs:

```text
node: 4
run_name: nl12_cms_s17
lines: 200
region: us-east-2
outputs_bucket: edullm-block-outputs-us-east-2
```

Training logs print JSON lines for losses and eval metrics. The key comparison
is the last eval row for each run.

## End Of Window

Run the drain report before the block closes:

```text
Block: drain the fleet before AWS takes it back
```

Inputs:

```text
stop_runs: false during normal monitoring
region: us-east-2
```

Only use `stop_runs: true` in the last few minutes of the window, because it
asks containers to shut down.

After jobs finish and containers are gone, stale claims can be released with:

```text
Block: give a node back
```

Use `nodes: all` or a comma-separated list. The workflow refuses to release a
claim whose container is still running.

## Reading The Result

CMS is a positive architecture signal if the final metrics show:

```text
cms > static_adapter
cms > single_memory
cms_no_dgd < cms
cms_no_retention < cms or worse stability
cms_no_momentum < cms or slower recovery
cms_no_self_reference <= cms
```

The result is not paper-grade until it is repeated over seeds. If the core run
wins, run:

```bash
python3 scripts/make_block_12h_configs.py --profile paper12
```

Then dispatch `WAVE=1`; after those runs finish and nodes are released, dispatch
`WAVE=2`.

If you can collect the `results/` directories back into one checkout, summarize
with:

```bash
python3 scripts/compare_nested_results.py --manifest configs/generated_block_12h/manifest.txt
```
