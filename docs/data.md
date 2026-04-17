# Data

## Overview

The dataset used in CETSAx–NADPH originates from a **Cellular Thermal Shift Assay (CETSA)**
study measuring the thermal stability of human proteome proteins in response to graded
concentrations of NADPH. Each row represents a single protein under a specific experimental
condition (replicate). The dose–response profile across ten NADPH concentrations is the
primary input to the curve-fitting stage.

The example data (`data/nadph.csv`) is derived from the publicly available dataset of
Dziekan *et al.* (Nature Protocols, 2020,
[DOI: 10.1038/s41596-020-0310-z](https://doi.org/10.1038/s41596-020-0310-z)).
The original `.Rdata` file was converted to `.csv` format for use in this pipeline.

!!! note "Using Your Own Data"
    **Any dataset that matches the schema described below will work with CETSAx.**
    You do not need to use the provided example file. Simply point `input_csv` in
    `config.yaml` to your own CSV file, and update `dose_columns` under
    `experiment:` to reflect your actual NADPH concentrations.

---

## File Format

The required input is a comma-separated values (`.csv`) file with the following structure:

- An optional index column (any column whose name starts with `Unnamed:` is automatically dropped).
- One **protein identifier** column.
- One **condition** column (typically distinguishes experimental replicates).
- Exactly **ten numeric dose columns** corresponding to the NADPH concentrations used.
- Three **quality-control** columns.

### Minimal Example

```text
"","id","description","condition","3.81e-06","1.526e-05","6.104e-05","0.00024414",
"0.00097656","0.00390625","0.015625","0.0625","0.25","1","sumUniPeps","sumPSMs","countNum"
"1","P04075","Fructose-bisphosphate aldolase A","NADPH.r1",1,0.97,0.99,0.97,...,3,1019,72
```

---

## Data Dictionary

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| *(index)* | integer | Optional row index — dropped automatically on load | `1` |
| `id` | string | UniProt accession or protein identifier | `P04075` |
| `description` | string | Human-readable protein description | `Fructose-bisphosphate aldolase A` |
| `condition` | string | Experimental condition or replicate label | `NADPH.r1`, `NADPH.r2` |
| `3.81e-06` | float | Normalised abundance at 3.81 µM NADPH | `1.000` |
| `1.526e-05` | float | Normalised abundance at 15.26 µM NADPH | `0.974` |
| `6.104e-05` | float | Normalised abundance at 61.04 µM NADPH | `0.991` |
| `0.00024414` | float | Normalised abundance at 244.14 µM NADPH | `0.966` |
| `0.00097656` | float | Normalised abundance at 976.56 µM NADPH | `0.967` |
| `0.00390625` | float | Normalised abundance at 3.906 mM NADPH | `0.978` |
| `0.015625` | float | Normalised abundance at 15.625 mM NADPH | `1.002` |
| `0.0625` | float | Normalised abundance at 62.5 mM NADPH | `0.984` |
| `0.25` | float | Normalised abundance at 250 mM NADPH | `1.046` |
| `1` | float | Normalised abundance at 1 M NADPH | `0.987` |
| `sumUniPeps` | integer | Number of unique peptides identified | `3` |
| `sumPSMs` | integer | Total peptide-spectrum matches | `1019` |
| `countNum` | integer | Number of quantified data points | `72` |

!!! note "Abundance normalisation"
    Dose columns contain **fold-change values** relative to the vehicle/lowest-dose
    condition. Values around 1.0 indicate no change; values > 1 indicate stabilisation
    and values < 1 indicate destabilisation.

---

## Quality Control Thresholds

Proteins are filtered before curve fitting using three thresholds defined in `config.yaml`:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `min_unique_peptides` | 3 | Minimum number of unique peptides required |
| `min_psms` | 15 | Minimum total PSMs required |
| `min_countnum` | 8 | Minimum quantified data points required |

Rows that fall below any threshold are excluded before downstream analysis.

---

## Notes & Warnings

!!! warning "Delimiter"
    The pipeline expects a **comma-separated** (`.csv`) file. Tab-separated or
    semicolon-separated files must be converted before use.

!!! warning "Missing dose columns"
    If any of the dose columns listed under `experiment.dose_columns` in `config.yaml`
    are absent from your CSV, the pipeline will raise a `KeyError` at the curve-fitting
    stage. Always verify that your column names match the config exactly.

!!! warning "Non-numeric dose values"
    Dose columns must contain numeric values. String values (e.g. `"N/A"`, empty cells)
    are coerced to `NaN` and may cause individual proteins to be excluded.

!!! note "Replicates"
    CETSAx is designed to work with **at least two replicates** per condition. The
    hit-calling stage uses replicate agreement to increase confidence. Single-replicate
    data can be analysed but will produce less reliable hit calls.

!!! tip "Large datasets"
    The provided example file contains ~12,000 rows. For larger proteomes, increase the
    parallelism in Snakemake (`--cores`) and consider enabling the embedding cache
    (`pooled_cache: true` in `config.yaml`) to avoid redundant ESM-2 forward passes.
