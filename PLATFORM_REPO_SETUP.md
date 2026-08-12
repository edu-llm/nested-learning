# Repository Setup For eduLLM

This directory is now shaped for both eduLLM paths:

- Capacity block workflows clone this public Git branch and run commands inside the fixed
  OLMo-core fleet image.
- The normal platform image path can build `.edullm/Dockerfile` after the repository is
  registered in `edu-llm/platform` and the platform-side IAM/ECR deploys have run.

## Installed CLI

The supported install command is:

```bash
uv tool install --force git+https://github.com/edu-llm/platform
edullm --version
```

This machine currently answers `edullm 4.5.0`.

## Git Branch

Use an `edullm/...` branch so the image workflow runs when this repo is pushed:

```bash
git push -u origin edullm/nested-learning-block-12h
```

The capacity block workflow can clone any public branch you name, but `edullm/**` is the
normal platform image-build convention.

## Capacity Block

The block does not pull this repository's custom image. It uses the already-launched
OLMo-core image, clones this branch to `/work`, and runs either the command field or
`.edullm/run.yaml`.

Recommended dispatch path:

```bash
REPOSITORY=edu-llm/nested-learning BRANCH=edullm/nested-learning-block-12h bash scripts/dispatch_block_12h_runs.sh
```

After checking the printed commands and confirming the nodes are free:

```bash
REPOSITORY=edu-llm/nested-learning BRANCH=edullm/nested-learning-block-12h EXECUTE=1 bash scripts/dispatch_block_12h_runs.sh
```

The helper uses `processes=1` because `scripts/run_block_12h_variant.sh` starts
`accelerate launch` inside the container.

## Platform Image

The caller workflow is `.github/workflows/build-research-image.yml`. It calls:

```text
edu-llm/platform/.github/workflows/build-research-image.yml@main
```

Before it can publish, the GitHub repository needs:

```text
Settings -> Secrets and variables -> Actions -> Variables
AWS_ECR_PUBLISHER_ROLE_ARN=<publisher role ARN>
```

That repository variable is set on `edu-llm/nested-learning`.

The platform registry also needs an entry for `nested-learning`; otherwise the reusable
workflow correctly refuses with `unregistered_repository`. Registration PR:

```text
https://github.com/edu-llm/platform/pull/448
```

## Registration

The automated registration dispatch exposed a platform workflow issue: the job used
`GITHUB_REPOSITORY_ID` as an environment variable, which collided with the Actions
runtime's own current-repository id. I pushed the generated registration manually as
`edu-llm/platform#448`.

For future repositories, the intended command is:

```bash
edullm add repository \
  --repository nested-learning \
  --dockerfile .edullm/Dockerfile \
  --reason "Nested Learning OLMoE/CMS architecture experiments on official MoE base models."
```

That command prepares a reviewed configuration change in `edu-llm/platform`; it does not
launch a run.
