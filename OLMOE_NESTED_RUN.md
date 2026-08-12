# OLMoE Nested Learning Experiment

This scaffold tests the Nested Learning claim at a higher scale on an official
MoE base model: `allenai/OLMoE-1B-7B-0125`.

The raw model config currently reports:

- `model_type`: `olmoe`
- hidden size: `2048`
- layers: `16`
- experts: `64`
- experts per token: `8`
- context length: `4096`

## Question

Does a multi-timescale memory adapter improve continual in-context learning
over matched controls on top of a strong official MoE base model?

## Variants

Run all three:

1. `olmoe_static_adapter_cluster.json`
   - Static SwiGLU adapters in the upper MoE blocks.
   - Controls for extra trainable parameters.

2. `olmoe_single_memory_cluster.json`
   - One online associative memory timescale.
   - Controls for test-time updating by itself.

3. `olmoe_cms_cluster.json`
   - Three online memory timescales: `128`, `512`, and `2048` token chunks.
   - Tests the Nested Learning / CMS hypothesis.

The default injection target is:

```text
^model\.layers\.\d+\.mlp$
```

For OLMoE, this wraps the MoE/MLP block inside decoder layers with a residual
adapter. By default, only layers `8-15` are wrapped.

## Data

The first version uses deterministic synthetic episodic fact streams:

```text
Fact: unit-1234-007 has code amber.
Update: unit-1234-007 now has code jade.
Question: What is the current code for unit-1234-007?
Answer: jade
```

Half the facts are volatile and receive updates. Half are stable and receive
rehearsal. This creates a direct test of update-speed tradeoffs:

- fast memory should help volatile facts,
- slow memory should help stable facts,
- CMS should win only if routing across timescales helps.

## Quick Smoke Run

```bash
bash scripts/run_olmoe_cms_smoke.sh
```

This downloads the model and runs a short adapter-only CMS training job. It is
meant to validate the environment and module patching, not to produce a paper
result.

## Full Ablation Run

On a SLURM cluster:

```bash
sbatch scripts/run_olmoe_ablation_array.slurm
```

Or run one variant:

```bash
CONFIG=configs/olmoe_cms_cluster.json sbatch scripts/run_olmoe_cms_cluster.slurm
```

## Compare Results

After the jobs finish:

```bash
python3 scripts/compare_nested_results.py --manifest configs/generated_paper_ablation/manifest.txt
```

## Success Criteria

The CMS variant is a real signal only if it beats both controls:

```text
CMS accuracy > static adapter accuracy
CMS accuracy > single-memory accuracy
CMS improves at least one of volatile/stable accuracy without collapsing the other
CMS effect appears in at least 3 seeds or repeated runs
```

For a stronger follow-up, duplicate the three cluster configs with different
`run.seed`, `data.seed`, and `data.eval_seed`.

You can generate a 3-variant x 3-seed matrix with:

```bash
python3 scripts/make_ablation_configs.py
sbatch scripts/run_generated_ablation_array.slurm
```

For the paper-aligned matrix, including DGD, momentum, retention,
self-reference, level-count, BPTT-through-update, base-ICL, and M3-lite
ablations:

```bash
python3 scripts/make_paper_ablation_configs.py
sbatch scripts/run_paper_ablation_array.slurm
```

See `NL_PAPER_AUDIT.md` for the full paper-to-run coverage table.

## 12-Hour GitHub Block Run

For the `edu-llm/platform` capacity-block lane, use the bounded matrix:

```bash
python3 scripts/make_block_12h_configs.py
```

This writes eight configs under `configs/generated_block_12h/`, one for each
node: base eval, static adapter, single-memory, CMS, and four CMS mechanism
ablations. See `GITHUB_BLOCK_12H_RUN.md` for the exact `Block:` workflow inputs,
including the print-only dispatch helper:

```bash
REPOSITORY=edu-llm/nested-learning BRANCH=edullm/nested-learning-block-12h bash scripts/dispatch_block_12h_runs.sh
```

## Important Caveats

This run does not prove that catastrophic forgetting is solved. It tests a
narrow architectural mechanism: whether multiple update frequencies can improve
online context compression in a mixed stable/volatile stream.

OLMoE has a 4096-token context window, so this is primarily a continual
in-context adaptation experiment, not a 128k long-context experiment. If the CMS
variant wins here, the next natural test is to replicate the same scaffold on a
long-context base such as Qwen3 or Mistral NeMo.
