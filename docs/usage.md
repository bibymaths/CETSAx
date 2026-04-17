# Usage

This page describes how to run CETSAx–NADPH end-to-end, what each stage does, and how to work with the outputs. The goal
is to let you move from raw CETSA data to biologically interpretable results with minimal friction.

---

## Overview

CETSAx is designed as a **pipeline-first system**. You typically do not run individual scripts manually. Instead, you
define your dataset and configuration, and let Snakemake orchestrate all steps.

The workflow follows a strict progression:

- Data → curve fitting → scoring → systems analysis → sequence modeling → interpretation

Each stage produces outputs that are consumed by the next stage. You can stop early if you only need part of the
analysis.

---

## Installation

### Clone the repository

```bash
git clone https://github.com/yourusername/cetsax-nadph.git
cd cetsax-nadph
```

---

### Environment setup (uv, recommended)

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

This is the fastest and cleanest way to set up dependencies.

---

### Alternative: Conda

```bash
conda env create -f environment.yml
conda activate cetsax
pip install -r requirements.txt
pip install -e .
```

Use this if you rely on Conda-based workflows or shared environments.

---

### Hardware notes

* CPU-only execution is sufficient for curve fitting and systems analysis.
* A **CUDA-capable GPU is strongly recommended** for sequence modeling (ESM-2).
* If no GPU is available, disable or skip sequence modeling steps.

---

## Input Data

The pipeline expects CETSA data in tabular form.

### Minimal structure

* `id`: protein identifier
* `condition`: experimental condition or replicate
* Dose columns: numeric values corresponding to concentrations

Example:

```text
id,condition,0.01,0.1,1,10,100
P12345,rep1,0.98,1.02,1.10,1.25,1.30
```

The exact column names for doses are defined in `config.yaml`.

---

## Configuration

All runtime behavior is controlled through `config.yaml`.

```yaml
input_csv: "data/nadph.csv"

python_bin: ".venv/bin/python"

epochs: 10
batch_size: 8
task: "classification"
device: "cuda"
```

### Key parameters

* `input_csv`: path to CETSA dataset
* `epochs`, `batch_size`: sequence model training
* `task`: `"classification"` or `"regression"`
* `device`: `"cuda"` or `"cpu"`

You rarely need to modify anything else for standard runs.

---

## Running the Pipeline

Execute the full workflow:

```bash
snakemake --cores 8
```

### What happens internally

Snakemake executes the pipeline in stages:

1. Curve fitting
2. Hit calling and QC
3. System-level analysis
4. Sequence dataset construction
5. Model training
6. Prediction and evaluation
7. Network analysis
8. Visualization

Each step is cached. If a step completes successfully, it will not be recomputed unless inputs change.

---

## Pipeline Stages (What Each Step Does)

### 1. Curve Fitting

* Fits a logistic model to each protein’s dose–response curve
* Outputs:

    * EC50
    * Hill coefficient
    * R²
    * Δmax

This is the quantitative backbone of the pipeline.

---

### 2. Hit Calling

* Filters unreliable fits
* Classifies proteins into:

    * strong responders
    * weak responders
    * non-responders

This step converts raw fits into interpretable biological categories.

---

### 3. System-Level Analysis

* PCA and clustering of response profiles
* Pathway enrichment (if annotations are available)

This step reveals coordinated behavior across proteins.

---

### 4. Sequence Dataset Construction

* Maps protein IDs to sequences (UniProt rescue if needed)
* Builds labeled dataset for ML

This is where experimental data becomes training data.

---

### 5. Sequence Model Training

* Uses ESM-2 embeddings
* Learns sequence determinants of NADPH response

Training modes include:

* pooled embeddings (fastest)
* residue representations (interpretability)
* token-level training (most flexible, slowest)

---

### 6. Prediction and Evaluation

* Generates predictions for all proteins
* Produces:

    * confusion matrix
    * ROC curves
    * probability distributions

---

### 7. Explainability

* Computes saliency maps
* Computes Integrated Gradients

Outputs are residue-level importance scores.

---

### 8. Network Analysis

* Builds co-stabilization network
* Detects modules of coordinated response

This provides system-level structure.

---

## Outputs

All results are stored under `results/`.

### Key directories

* `results/hit_calling/` → fit results and classifications
* `results/system_analysis/` → PCA, clustering
* `results/network/` → co-stabilization graphs
* `results/plots/` → model evaluation and interpretability
* `results/detailed_plots/` → per-protein curves

---

## How to Work With Results

### Curve fits

* Use EC50 and Δmax to assess biochemical relevance
* Check R² before trusting any result

---

### Hit tables

* Focus on strong responders
* Look for pathway-level enrichment

---

### PCA and clustering

* Identify response modes
* Detect groups of functionally related proteins

---

### Sequence model outputs

* Evaluate model performance (ROC, confusion matrix)
* Inspect misclassifications for interesting biology

---

### Interpretability outputs

* Map important residues to:

    * domains
    * binding sites
    * motifs

This is where the model becomes biologically useful.

---

### Network outputs

* Identify modules
* Connect proteins through shared response patterns

---

## Running Individual Steps (Optional)

You can run specific parts of the pipeline:

```bash
snakemake results/hit_calling/
snakemake results/network/
```

This is useful for debugging or iterative development.

---

## Common Issues

### Pipeline stops early

* Check missing dependencies
* Verify `config.yaml` paths

---

### Poor curve fits

* Data may be noisy or non-sigmoidal
* Increase QC thresholds or inspect raw curves

---

### Sequence model fails

* Likely GPU memory issue
* Reduce batch size or switch to pooled mode

---

### Empty outputs

* Usually due to strict filtering
* Relax thresholds in configuration

---

## Recommended Workflow

For first-time users:

* Run full pipeline
* Inspect `results/hit_calling/`
* Move to `system_analysis/`
* Then evaluate sequence model
* Finally, interpret saliency and network results

Avoid jumping directly to deep learning outputs without validating upstream steps.

---

## Summary

The intended usage pattern is:

* Run once → inspect outputs → refine parameters → rerun selectively

CETSAx is designed to be iterative, not one-shot. The more you align the pipeline with your biological question, the
more meaningful the results become.
