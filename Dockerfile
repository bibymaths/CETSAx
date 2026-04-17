# CETSAx – Snakemake workflow + Python package image
# Builds a lean, reproducible image suitable for GHCR publishing and local runs.
#
# CUDA notes
# ----------
# PyTorch 2.9.x ships all CUDA shared libraries as PyPI wheels (nvidia-cublas-cu12, etc.;
# see requirements.txt). No system CUDA toolkit is required at build OR install time.
# GPU access at runtime still requires --gpus all and a CUDA-capable driver on the host.
#
# tokenizers note
# ---------------
# tokenizers==0.22.1 ships as a pre-built cp39-abi3-manylinux_2_17_x86_64 wheel.
# Rust is NOT required in this container.
#
# ESM-2 model weights
# -------------------
# fair-esm uses torch.hub (torch.hub.load_state_dict_from_url) to download weights from
# https://dl.fbaipublicfiles.com/fair-esm/models/<model>.pt at first use.
# The download location is controlled by the TORCH_HOME env var (set below to /app/models).
# Mount a host directory at /app/models to cache weights across runs, e.g.:
#   -v ./models:/app/models

FROM python:3.11-slim

# git is required by gitpython, which snakemake depends on at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Bring in the uv installer (statically linked binary, no extra deps)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests AND README.md before source so Docker can cache this layer.
# README.md is required by setuptools (pyproject.toml: readme = "README.md") when
# building the cetsax package metadata during `pip install -e .`.
COPY pyproject.toml uv.lock requirements.txt README.md ./

# Install all pinned runtime dependencies into the system Python interpreter.
# --no-cache keeps the layer lean; --system avoids an extra venv inside the container.
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the workflow sources
COPY cetsax/    ./cetsax/
COPY scripts/   ./scripts/
COPY workflow/  ./workflow/
COPY Snakefile  config.yaml ./

# Install the cetsax package itself (dependencies already satisfied above)
RUN uv pip install --system --no-cache --no-deps -e .

# Placeholder directories – real data/results are expected to be mounted as volumes.
# /app/models is the torch.hub cache dir for ESM-2 weights; mount a host volume here
# to avoid re-downloading weights on every container start.
RUN mkdir -p data results models

# TORCH_HOME controls where torch.hub (used by fair-esm) downloads and caches model
# weights.  Setting it to /app/models makes the cache path predictable and mountable.
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/models

# Default: execute the full Snakemake pipeline using the bundled local profile.
#
# CPU-only run (no GPU, no ESM forward pass required when using cached embeddings):
#   docker run --rm \
#     -v ./data:/app/data \
#     -v ./results:/app/results \
#     -v ./models:/app/models \
#     ghcr.io/bibymaths/cetsax
#
# GPU run (required for train_model / predict rules that run ESM-2 forward):
#   docker run --rm --gpus all \
#     -v ./data:/app/data \
#     -v ./results:/app/results \
#     -v ./models:/app/models \
#     ghcr.io/bibymaths/cetsax
CMD ["snakemake", "--cores", "all", "--snakefile", "Snakefile", "--profile", "workflow/profiles/local"]
