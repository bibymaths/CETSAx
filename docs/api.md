# API Reference

This page documents the public Python API of `cetsax`. The package is organized around the analytical flow of the
project: configuration and data access, curve fitting, sensitivity scoring, systems-level analysis, sequence learning,
and visualization.

The API is intentionally modular. Most functions operate on `pandas.DataFrame` objects with a shared schema built around
protein identifiers, experimental conditions, and dose columns. In practice, this means individual modules can be used
independently, but they are designed to compose into a coherent pipeline.

## Package layout

- `cetsax.config` defines shared constants and configuration loading.
- `cetsax.dataio` handles data ingestion and QC.
- `cetsax.models` contains the core ITDR response model.
- `cetsax.fit` performs curve fitting.
- `cetsax.hits` and `cetsax.viz_hits` handle hit calling and diagnostics.
- `cetsax.sensitivity` computes protein-level sensitivity metrics.
- `cetsax.enrichment`, `cetsax.network`, `cetsax.redox`, `cetsax.latent`, and `cetsax.mixture` support systems-level
  interpretation.
- `cetsax.ml`, `cetsax.bayes`, and `cetsax.deeplearn.*` provide statistical and learning-based extensions.
- `cetsax.plotting`, `cetsax.viz`, and `cetsax.viz_predict` provide plotting and result interpretation.

## Core package

### `cetsax`

::: cetsax
options:
show_root_heading: true
show_root_toc_entry: false
show_source: false
members_order: source

## Configuration and data I/O

### `cetsax.config`

This module exposes the shared experimental schema. Most downstream modules rely on the constants defined here, so this
is effectively the contract layer of the package.

::: cetsax.config
options:
show_root_heading: true
show_source: false
members_order: source

### `cetsax.dataio`

This module loads CETSA input tables and applies the first QC filter before any modeling is attempted.

::: cetsax.dataio
options:
show_root_heading: true
show_source: false
members_order: source

### `cetsax.annotate`

This module supports annotation and sequence retrieval workflows, including ID normalization, MyGene-based annotation,
and UniProt FASTA fetching.

::: cetsax.annotate
options:
show_root_heading: true
show_source: false
members_order: source

## Mathematical modeling and fitting

### `cetsax.models`

This module contains the canonical ITDR model function used for CETSA dose–response evaluation.

::: cetsax.models
options:
show_root_heading: true
show_source: false
members_order: source

### `cetsax.fit`

This is the main fitting engine. It handles monotonic smoothing, bounded optimization, fit diagnostics, and bulk fitting
across proteins and conditions.

::: cetsax.fit
options:
show_root_heading: true
show_source: false
members_order: source

### `cetsax.bayes`

This module provides Bayesian alternatives for parameter inference when posterior uncertainty is of interest rather than
only point estimates.

::: cetsax.bayes
options:
show_root_heading: true
show_source: false
members_order: source

## Hit calling and sensitivity scoring

### `cetsax.hits`

This module provides direct hit filtering and hit summarization from fitted parameters.

::: cetsax.hits
options:
show_root_heading: true
show_source: false
members_order: source

### `cetsax.sensitivity`

This module computes the NADPH Sensitivity Score and related protein-level summary metrics.

::: cetsax.sensitivity
options:
show_root_heading: true
show_source: false
members_order: source

### `cetsax.viz_hits`

This module combines hit-calling logic with diagnostic visualization utilities and export helpers.

::: cetsax.viz_hits
options:
show_root_heading: true
show_source: false
members_order: source

## Systems-level analysis

### `cetsax.enrichment`

This module implements pathway-level summarization and both binary and continuous enrichment testing.

::: cetsax.enrichment
options:
show_root_heading: true
show_source: false
members_order: source

### `cetsax.network`

This module derives co-stabilization networks from CETSA response profiles and detects response modules.

::: cetsax.network
options:
show_root_heading: true
show_source: false
members_order: source

### `cetsax.redox`

This module constructs interpretable redox axes and pathway-level redox summaries from sensitivity, hit, and optional
network information.

::: cetsax.redox
options:
show_root_heading: true
show_source: false
members_order: source

### `cetsax.latent`

This module builds feature matrices and fits low-dimensional latent representations such as PCA and factor analysis.

::: cetsax.latent
options:
show_root_heading: true
show_source: false
members_order: source

### `cetsax.mixture`

This module supports Gaussian mixture modeling for response-state discovery and soft cluster assignment.

::: cetsax.mixture
options:
show_root_heading: true
show_source: false
members_order: source

### `cetsax.ml`

This module contains classical machine-learning utilities for curve feature extraction, clustering, and outlier
detection.

::: cetsax.ml
options:
show_root_heading: true
show_source: false
members_order: source

## Visualization

### `cetsax.plotting`

This module provides direct plotting helpers for raw curves and global fit diagnostics.

::: cetsax.plotting
options:
show_root_heading: true
show_source: false
members_order: source

### `cetsax.viz`

This module provides higher-level visualization for pathway effects, redox axes, latent structure, and mixture-model
outputs.

::: cetsax.viz
options:
show_root_heading: true
show_source: false
members_order: source

### `cetsax.viz_predict`

This module focuses on downstream evaluation of predictive models, including confusion matrices, ROC curves,
saliency-style plots, and biologically oriented diagnostic summaries.

::: cetsax.viz_predict
options:
show_root_heading: true
show_source: false
members_order: source

## Deep learning

### `cetsax.deeplearn`

This subpackage contains sequence-based learning modules built around protein language models and custom training
utilities.

::: cetsax.deeplearn
options:
show_root_heading: true
show_source: false
members_order: source

### `cetsax.deeplearn.esm_seq_nadph`

This module contains the ESM-based training, caching, inference, and explainability workflow.

::: cetsax.deeplearn.esm_seq_nadph
options:
show_root_heading: true
show_source: false
members_order: source

### `cetsax.deeplearn.my_seq_nadph`

This module contains the Transformers/Hugging Face based sequence-modeling workflow with support for pooled,
residue-level, and token-level training modes.

::: cetsax.deeplearn.my_seq_nadph
options:
show_root_heading: true
show_source: false
members_order: source

## Notes on expected data structures

Most analytical functions in `cetsax` assume one of a few standard tabular forms.

- Raw CETSA tables usually include:
    - `id`
    - `condition`
    - dose columns defined in `cetsax.config.DOSE_COLS`

- Fit tables usually include:
    - `EC50`
    - `log10_EC50`
    - `Hill`
    - `R2`
    - `delta_max`

- Protein-level summary tables usually include:
    - `id`
    - sensitivity metrics such as `NSS`
    - optional annotations such as pathway or redox role