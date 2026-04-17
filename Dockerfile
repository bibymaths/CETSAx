# CETSAx – Snakemake workflow + Python package image
# Builds a lean, reproducible image suitable for GHCR publishing and local runs.
#
# Runtime note:
#   PyTorch 2.9.x ships its own CUDA libraries as PyPI wheels (nvidia-cublas-cu12, etc.),
#   so no system CUDA toolkit is required.  A base Python slim image is sufficient.
#
# GPU access at runtime still requires --gpus all (or equivalent) on the host.

FROM python:3.11-slim

# git is required by gitpython, which snakemake depends on at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Bring in the uv installer (statically linked binary, no extra deps)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifests before source so Docker can cache this layer
COPY pyproject.toml uv.lock requirements.txt ./

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

# Placeholder directories – real data/results are expected to be mounted as volumes
RUN mkdir -p data results

ENV PYTHONUNBUFFERED=1

# Default: execute the full Snakemake pipeline using the bundled local profile.
# Mount data at /app/data and results at /app/results, e.g.:
#   docker run --rm -v ./data:/app/data -v ./results:/app/results ghcr.io/bibymaths/cetsax
CMD ["snakemake", "--cores", "all", "--snakefile", "Snakefile", "--profile", "workflow/profiles/local"]
