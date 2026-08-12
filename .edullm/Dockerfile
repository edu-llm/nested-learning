# Research image for the normal eduLLM platform image path.
#
# The platform reusable workflow passes the registered, digest-pinned base image as
# --build-arg BASE_IMAGE and refuses hardcoded FROM images. Keep the ARG bare and global.
ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/opt/nested-learning

WORKDIR /opt/nested-learning

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends gcc g++; \
    rm -rf /var/lib/apt/lists/*; \
    cc --version

COPY requirements-olmoe.txt ./
RUN set -eux; \
    python -m pip install --upgrade pip; \
    python -m pip install --no-cache-dir -r requirements-olmoe.txt; \
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); assert torch.version.cuda is not None, 'eduLLM GPU images must carry CUDA torch'"

COPY . ./
RUN set -eux; \
    python -m py_compile train_olmoe_nested.py nested_olmoe/*.py scripts/make_block_12h_configs.py scripts/compare_nested_results.py

CMD ["bash", "scripts/run_block_12h_variant.sh", "configs/generated_block_12h/cms_seed17.json"]
