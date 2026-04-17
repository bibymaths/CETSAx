# cetsax — Comprehensive Improvement Prompt

## Role & Constraints

You are a senior bioinformatics software engineer working on **cetsax**, a novel 
Snakemake-orchestrated bioinformatics pipeline. The project is **fully functional**. 
Your prime directive:

> **Everything that works must continue to work. Nothing breaks. No regressions.**

Before changing any file, read it fully. Before editing `config.yml` or any 
Snakemake `.smk` file, trace every rule dependency. If you are uncertain whether 
a change is safe, **ask before proceeding**. Do not batch-apply changes without 
pausing at each phase checkpoint for confirmation.

Author for all generated/modified files:
- Name: **Abhinav Mishra**
- SPDX identifier: `SPDX-FileCopyrightText: 2024 Abhinav Mishra`
- License: match the existing project license from `LICENSE` or `pyproject.toml`

---

## Architecture Awareness (Read First)

1. **Orchestration**: Snakemake controls the entire pipeline. `config.yml` is the 
   single source of truth — CSV input path, process flags, parameters, and output 
   paths are all controlled from there. **Do not change `config.yml` structure.**
2. **Docker builds are triggered by Snakemake rules** — any Dockerfile or 
   docker-compose change must remain compatible with existing Snakemake `rule` 
   `container:` directives or `singularity:` wrappers.
3. **Sequence models**: ESM2 protein models from Facebook/Meta (`facebook/esm2_*`) 
   are used. The project was tested with the most basic ESM2 model on a Tesla T4 
   16 GB GPU due to resource constraints.
4. **Python targets**: 3.10, 3.11, 3.12.

---

## Phase 1 — Test Suite Fixes (Highest Priority)

**Goal**: All tests pass on Python 3.10, 3.11, and 3.12 in CI.

- [ ] Audit `tests/` for any syntax, import, or API incompatibilities across 
  Python 3.10, 3.11, 3.12.
- [ ] Fix deprecated `datetime.utcnow()` (use `datetime.now(timezone.utc)`), 
  removed `distutils`, `typing` aliases (`Union` → `X | Y` for 3.10+), 
  and any `collections.abc` import issues.
- [ ] Add a `conftest.py` at the project root with:
  - Shared fixtures
  - Path patching for `data/` directory references
  - A `pytest.ini` or `pyproject.toml` `[tool.pytest.ini_options]` section 
    specifying `testpaths`, `python_files`, and `filterwarnings`
- [ ] Do **not** change test logic — only fix compatibility issues.

**Checkpoint**: Run `pytest` and confirm green on all three Python versions before 
proceeding.

---

## Phase 2 — Ruff Linting & GitHub Workflow

**Goal**: Clean, consistently linted codebase with automated enforcement.

- [ ] Add `ruff` to dev dependencies in `pyproject.toml` (or `setup.cfg`).
- [ ] Create `ruff.toml` (or `[tool.ruff]` in `pyproject.toml`) with:
  ```toml
  [tool.ruff]
  target-version = "py310"
  line-length = 88
  select = ["E", "F", "W", "I", "UP", "B", "C4", "SIM"]
  ignore = ["E501"]  # adjust per project style
  
  [tool.ruff.isort]
  known-first-party = ["cetsax"]  # adjust to actual package name
  ```
- [ ] Run `ruff check --fix` on the entire codebase. Fix each file carefully:
  - Preserve all biological/scientific logic exactly.
  - Do not auto-remove "unused imports" that are re-exported or used in type hints.
  - Review every auto-fix diff before applying.
- [ ] Add `.github/workflows/lint.yml`:
  ```yaml
  name: Lint
  on: [push, pull_request]
  jobs:
    ruff:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: astral-sh/ruff-action@v1
          with:
            args: "check ."
  ```

---

## Phase 3 — Docstrings & Script Headers

**Goal**: Every public function, class, and CLI script is fully documented with 
SPDX identifiers.

- [ ] Add SPDX headers to every `.py` file that lacks one:
  ```python
  # SPDX-FileCopyrightText: 2024 Abhinav Mishra
  # SPDX-License-Identifier: <LICENSE-IDENTIFIER>
  ```
- [ ] Write NumPy-style docstrings for:
  - All public functions and methods
  - All classes
  - All Snakemake helper/wrapper scripts
- [ ] For CLI scripts (`scripts/`, `bin/`, or `workflow/scripts/`), add a module-level 
  docstring:
  ```python
  """
  script_name.py — One-line summary.

  Description:
      Longer description of what this script does in the pipeline context.

  Usage:
      Called by Snakemake rule `rule_name` via config.yml.

  Author:
      Abhinav Mishra
  """
  ```
- [ ] Do not alter any logic while adding docstrings.

---

## Phase 4 — `docs/data.md`

**Goal**: A complete, accurate data documentation page.

- [ ] Inspect every `.csv` file in `data/` — note column names, dtypes, row counts, 
  and biological meaning from context.
- [ ] Inspect the existing `data/README` or `data/readme.md` (if present).
- [ ] Create `docs/data.md` with sections:
  - **Overview** — what the data represents biologically
  - **File Format** — column-by-column description of the required CSV schema
  - **Using Your Own Data** — note that *any* data following the schema of the 
    provided CSV files in `data/` will work. Parameters and input paths are 
    controlled via `config.yml`.
  - **Data Dictionary** — table of column name, type, description, example value
  - **Notes & Warnings** — MkDocs `!!! warning` / `!!! note` admonitions for 
    common mistakes (wrong delimiter, missing columns, etc.)
- [ ] After creating `docs/data.md`, delete `data/README` / `data/readme.md`.

---

## Phase 5 — Sequence Models Documentation Update

**Goal**: Clarify ESM2 model flexibility in relevant docs section.

- [ ] Find the documentation section discussing sequence models (search for `ESM`, 
  `esm2`, `protein model`, or `embedding` in `docs/`).
- [ ] Add a subsection or admonition:
  ```markdown
  !!! info "ESM2 Model Flexibility"
      Any ESM2 protein model from [Facebook/Meta on HuggingFace](https://huggingface.co/facebook) 
      can be used (e.g., `facebook/esm2_t6_8M_UR50D` through `facebook/esm2_t48_15B_UR50D`).
      
      The most basic model (`esm2_t6_8M_UR50D`) was used during development due to 
      GPU resource constraints. Testing was performed on a **Tesla T4 (16 GB VRAM)**. 
      Larger models will require more VRAM — scale your model selection accordingly 
      via `config.yml`.
  ```

---

## Phase 6 — Figures into Documentation

**Goal**: All `docs/assets/figures/*.png` files are embedded at their most 
contextually meaningful location in the docs.

- [ ] List all `.png` files in `docs/assets/figures/`.
- [ ] For each image, infer its content from its filename and match it to the most 
  relevant section of the docs.
- [ ] Embed using MkDocs-compatible syntax:
  ```markdown
  ![Description of figure](assets/figures/filename.png)
  /// caption
  Figure N: Descriptive caption explaining what the figure shows.
  ///
  ```
- [ ] Do not move image files — use relative paths from the `.md` file location.

---

## Phase 7 — MkDocs Documentation Polish

**Goal**: Docs that delight both biologists and developers.

- [ ] Add MkDocs Material admonitions throughout:
  - `!!! note` for key concepts
  - `!!! warning` for pitfalls and common mistakes
  - `!!! tip` for shortcuts and best practices
  - `!!! danger` for irreversible operations
  - `!!! example` for usage examples with code blocks
- [ ] Add collapsible sections for long parameter tables:
  ```markdown
  ??? info "All config.yml parameters"
      | Parameter | Default | Description |
      |-----------|---------|-------------|
      ...
  ```
- [ ] Add tabbed content for multi-language or multi-platform instructions:
  ```markdown
  === "Local"
      ...
  === "HPC (SLURM)"
      ...
  === "Docker"
      ...
  ```
- [ ] Ensure all code blocks have language specifiers (` ```python`, ` ```yaml`, etc.)
- [ ] Add a report output subsection wherever pipeline outputs are discussed:
  ```markdown
  ### `report.html` — Integrated Results Overview
  The `report.html` file is a self-contained HTML report generated by Snakemake 
  that weaves all pipeline results — figures, tables, and metrics — into a single 
  interactive overview. It is the recommended first point of inspection after a 
  successful pipeline run.
  ```

---

## Phase 8 — `.zenodo.json`

**Goal**: Correct, complete Zenodo metadata for citation.

- [ ] Read `docs/citation.md` in full to extract: title, authors, description, 
  keywords, license, DOI (if present), and related identifiers.
- [ ] Create `.zenodo.json` at the project root following the Zenodo schema:
  ```json
  {
    "title": "<from citation.md>",
    "description": "<abstract from citation.md>",
    "upload_type": "software",
    "license": "<SPDX license identifier>",
    "creators": [
      {
        "name": "Mishra, Abhinav",
        "affiliation": "<from citation.md if present>",
        "orcid": "<from citation.md if present>"
      }
    ],
    "keywords": ["bioinformatics", "snakemake", "protein", "ESM2"],
    "related_identifiers": [],
    "access_right": "open"
  }
  ```
- [ ] Cross-reference every field with `citation.md` — do not invent metadata.

---

## Phase 9 — `.github/CODEOWNERS`

**Goal**: Automatic review assignment.

- [ ] Create `.github/CODEOWNERS`:
  ```
  # Default owner for everything
  *                    @abhinavmishra14
  
  # Snakemake workflow
  workflow/            @abhinavmishra14
  Snakefile            @abhinavmishra14
  config.yml           @abhinavmishra14
  
  # Documentation
  docs/                @abhinavmishra14
  
  # CI/CD
  .github/             @abhinavmishra14
  ```
  Replace `@abhinavmishra14` with your actual GitHub username.

---

## Phase 10 — Legal Files

**Goal**: Add `NOTICE` and `TERMS_OF_USE.md`.

- [ ] Create `NOTICE`:
  ```
  cetsax
  Copyright 2024 Abhinav Mishra
  
  This product includes software developed by Abhinav Mishra.
  [List any third-party components and their licenses here.]
  ```
- [ ] Create `TERMS_OF_USE.md` appropriate for a research software tool:
  - Academic/research use encouraged
  - Attribution required (cite the paper/zenodo DOI)
  - No warranty clause
  - Contact information for commercial licensing inquiries

---

## Phase 11 — Docker & Compose (ghcr.io)

**Goal**: Full container lifecycle scripts compatible with Snakemake's existing 
container directives.

**Critical**: Read all existing Snakemake rules that reference `container:` or 
`singularity:` before modifying any Dockerfile. The container image name and tag 
used in Snakemake rules must not change unless you update all rules atomically.

- [ ] Create `.dockerignore`:
  ```
  .git
  .github
  __pycache__
  *.pyc
  *.pyo
  .pytest_cache
  .ruff_cache
  htmlcov
  .coverage
  data/
  results/
  logs/
  .snakemake/
  ```
- [ ] Create `docker-compose.yml` (production, using `ghcr.io` image):
  ```yaml
  version: "3.9"
  services:
    cetsax:
      image: ghcr.io/<github-org-or-user>/cetsax:latest
      volumes:
        - ./data:/app/data:ro
        - ./results:/app/results
        - ./config.yml:/app/config.yml:ro
      environment:
        - PYTHONUNBUFFERED=1
  ```
- [ ] Create `docker-compose.dev.yml` (build from local source):
  ```yaml
  version: "3.9"
  services:
    cetsax:
      build:
        context: .
        dockerfile: Dockerfile
      volumes:
        - .:/app
        - ./data:/app/data
      environment:
        - PYTHONUNBUFFERED=1
        - DEVELOPMENT=1
  ```
- [ ] Create shell scripts with `#!/usr/bin/env bash` and `set -euo pipefail`:
  - `docker_up.sh` — `docker compose up -d`
  - `docker_down.sh` — `docker compose down`
  - `docker_interactive.sh` — `docker compose run --rm cetsax bash`
  - `docker_push.sh` — builds and pushes to `ghcr.io` using `GITHUB_TOKEN`
  - `docker_test_local.sh` — runs `pytest` inside the container
  - Make all scripts executable: `chmod +x docker_*.sh`

---

## Phase 12 — Pre-commit Config

**Goal**: Automated local quality gates.

- [ ] Create `.pre-commit-config.yaml`:
  ```yaml
  repos:
    - repo: https://github.com/pre-commit/pre-commit-hooks
      rev: v4.6.0
      hooks:
        - id: trailing-whitespace
        - id: end-of-file-fixer
        - id: check-yaml
        - id: check-json
        - id: check-merge-conflict
        - id: check-added-large-files
          args: ["--maxkb=5000"]
    - repo: https://github.com/astral-sh/ruff-pre-commit
      rev: v0.4.4
      hooks:
        - id: ruff
          args: [--fix]
        - id: ruff-format
  ```

---

## Phase 13 — Unified Release Workflow

**Goal**: Single `.github/workflows/release.yml` that handles PyPI, Docker (ghcr.io), 
and GitHub Release on `v*` tag push.

```yaml
name: Release

on:
  push:
    tags:
      - "v*"

permissions:
  contents: write
  packages: write
  id-token: write

jobs:
  test:
    uses: ./.github/workflows/test.yml  # reuse existing test workflow

  pypi-release:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install build
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}

  docker-release:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/metadata-action@v5
        id: meta
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=raw,value=latest
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}

  github-release:
    needs: [pypi-release, docker-release]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: softprops/action-gh-release@v2
        with:
          generate_release_notes: true
```

---

## Phase 14 — Snakemake HPC Scaling

**Goal**: HPC-ready Snakemake profiles without breaking local or Docker execution.

**Read `config.yml` fully before making any changes.** Do not alter rule logic.

- [ ] Create `workflow/profiles/slurm/config.yaml`:
  ```yaml
  executor: cluster-generic
  cluster-generic-submit-cmd: >
    sbatch
    --job-name={rule}
    --output=logs/slurm/{rule}_{wildcards}.out
    --error=logs/slurm/{rule}_{wildcards}.err
    --time={resources.runtime}
    --mem={resources.mem_mb}M
    --cpus-per-task={threads}
    --partition={resources.partition}
  cluster-generic-cancel-cmd: scancel
  cluster-generic-status-cmd: squeue --job {jobid} --noheader --format=%T
  jobs: 100
  latency-wait: 60
  retries: 2
  ```
- [ ] Create `workflow/profiles/local/config.yaml` (for local/Docker runs):
  ```yaml
  cores: all
  jobs: 1
  latency-wait: 5
  use-conda: false
  ```
- [ ] Add `resources:` directives to each Snakemake rule that lacks them:
  ```python
  resources:
      mem_mb=8000,
      runtime="2h",
      partition="gpu"  # only for GPU rules
  ```
  **Only add resources to rules — do not change rule inputs/outputs/params.**
- [ ] Add a `--profile` note to the docs:
  - Local: `snakemake --profile workflow/profiles/local`
  - SLURM HPC: `snakemake --profile workflow/profiles/slurm`
  - Docker: existing invocation unchanged

---

## Phase 15 — Pipeline Flow Diagram

**Goal**: A clear, visually appealing Mermaid flow diagram showing the full 
Snakemake orchestration.

- [ ] Read the `Snakefile` and all included `.smk` files to understand the DAG.
- [ ] Create a Mermaid diagram covering:
  - Input: CSV file (via `config.yml`)
  - All major rule stages (preprocessing → feature extraction → ESM2 embedding 
    → model training/inference → output generation → report)
  - Docker container boundaries (which rules run in containers)
  - Output artifacts
- [ ] Embed in `docs/workflow.md` (or the most appropriate existing docs page):
  ```markdown
  ## Pipeline Overview
  
  The following diagram shows the complete orchestration of the `cetsax` pipeline,
  from raw CSV input to final outputs, as managed by Snakemake.
  
  ```mermaid
  flowchart TD
      A[("config.yml\n(CSV path, params)")] --> B[/Input CSV/]
      B --> C[Rule: preprocess]
      C --> D[Rule: feature_extract]
      D --> E[Rule: esm2_embed\n🐳 Docker container]
      E --> F[Rule: train / infer]
      F --> G[Rule: evaluate]
      G --> H[/results/]
      H --> I[Rule: report]
      I --> J[("report.html\nIntegrated Overview")]
      
      style E fill:#dbeafe,stroke:#2563eb
      style J fill:#dcfce7,stroke:#16a34a
  ` ` `
  ```
  Update all rule names to match the actual Snakefile rules.

---

## Final Checklist (Run Before Closing PR)

- [ ] `pytest` passes on Python 3.10, 3.11, 3.12
- [ ] `ruff check .` returns zero errors
- [ ] `snakemake --dry-run` completes without errors (local profile)
- [ ] `docker compose -f docker-compose.dev.yml build` succeeds
- [ ] All docs pages render without broken image links (`mkdocs build --strict`)
- [ ] `.zenodo.json` validates at https://zenodo.org/deposit/validate
- [ ] No `config.yml` keys were renamed or removed
- [ ] All Snakemake rule `input:`/`output:`/`params:` blocks are unchanged

---

## Do Not Touch

- `config.yml` — structure and keys are frozen
- Existing Snakemake rule logic and `input:`/`output:` declarations
- Trained model weights or checkpoint files
- Any file in `data/` (except deleting README after Phase 4)
```

***

## How to Use This Prompt

1. Save the block above as `.github/prompts/cetsax-improvements.prompt.md` in your repo [].
2. Open GitHub Copilot Chat in Agent Mode (VS Code or github.com).
3. Type: `Use the prompt file .github/prompts/cetsax-improvements.prompt.md` and attach the relevant files for each phase.
4. Run **one phase at a time** — confirm the checkpoint passes before asking Copilot to proceed to the next [].
5. For the HPC phase (Phase 14), explicitly open the `Snakefile` and all `.smk` files as context tabs before invoking Copilot so it has full rule awareness [].

> **Pro tip**: The most effective way to use Copilot Agent Mode for novel research code is to give it explicit "do not touch" constraints alongside the tasks — this prevents the agent from over-reaching into working scientific logic while fixing surrounding infrastructure [].