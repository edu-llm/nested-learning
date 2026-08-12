# Nested Learning OLMoE Experiments

This repository tests the Nested Learning / convergent memory system idea on
the official `allenai/OLMoE-1B-7B-0125` MoE base model.

## Capacity Block

Generate the bounded 12-hour matrix:

```bash
python3 scripts/make_block_12h_configs.py
```

Print the `edu-llm/platform` Block dispatch commands:

```bash
REPOSITORY=owner/repo BRANCH=edullm/nested-learning-block-12h bash scripts/dispatch_block_12h_runs.sh
```

See `GITHUB_BLOCK_12H_RUN.md` for the full launch, status, logs, drain, and release runbook.

## Local Proof Of Concept

```bash
python3 continual_memory_poc.py
```

## Higher-Scale OLMoE Runs

```bash
bash scripts/run_olmoe_cms_smoke.sh
python3 scripts/make_paper_ablation_configs.py
sbatch scripts/run_paper_ablation_array.slurm
```

See `OLMOE_NESTED_RUN.md` and `NL_PAPER_AUDIT.md` for the paper-to-experiment mapping.

## eduLLM Platform

The repository includes:

- `.edullm/run.yaml` for a default capacity-block command.
- `.edullm/Dockerfile` for the normal eduLLM research image build path.
- `.github/workflows/build-research-image.yml` calling the platform reusable image workflow.

See `PLATFORM_REPO_SETUP.md` for registration and push notes.

