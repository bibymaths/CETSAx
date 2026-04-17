# Pipeline Overview

CETSAx is orchestrated entirely by Snakemake. The diagram below shows the complete
dependency graph — from raw CSV input to final outputs — as defined in the `Snakefile`.

## Pipeline Flow Diagram

The following diagram shows the complete orchestration of the `cetsax` pipeline,
from raw CSV input to final outputs, as managed by Snakemake.

```mermaid
flowchart TD
    A[("config.yaml\n(CSV path, params)")] --> B[/data/nadph.csv/]
    B --> C[Rule: fit_curves\nEC50, Hill, Δmax, R²]
    C --> D[Rule: hit_calling\nClassify responders]
    C --> E[Rule: annotate\nFetch sequences → FASTA]
    D --> E
    C --> F[Rule: system_analysis\nPCA, clustering, enrichment]
    D --> F
    E --> F
    F --> G[Rule: train_model\nESM-2 embedding + head\n🐳 GPU rule]
    C --> G
    E --> G
    G --> H[Rule: predict\nESM-2 inference + saliency\n🐳 GPU rule]
    E --> H
    H --> I[Rule: visualize\nPlots 1–14]
    G --> I
    C --> I
    E --> I
    I --> J[Rule: network_analysis\nCo-stabilization graph]
    B --> J
    J --> K[Rule: curve_ml\nK-means curve typing]
    B --> K
    K --> L[Rule: bayesian_fit\nMCMC EC50 posteriors]
    B --> L
    D --> L
    L --> M[Rule: detailed_plots\nPer-protein curves + GoF]
    B --> M
    C --> M
    D --> M
    G --> N[Rule: model_performance\nClassification report]
    H --> N
    I --> O[("results/plots/\nModel evaluation figures")]
    J --> P[("results/network/\nGEXF graph + modules")]
    K --> Q[("results/curve_ml/\nClusters + outliers")]
    L --> R[("results/bayesian/\nEC50 posteriors")]
    M --> S[("results/detailed_plots/\nPer-protein curves")]
    N --> T[("results/model_performance_report.txt")]
    O & P & Q & R & S & T --> U[("report.html\nIntegrated Results Overview")]

    style G fill:#dbeafe,stroke:#2563eb
    style H fill:#dbeafe,stroke:#2563eb
    style U fill:#dcfce7,stroke:#16a34a
```

!!! note "Running the pipeline"
    Run locally:
    ```bash
    snakemake --profile workflow/profiles/local
    ```
    Run on SLURM HPC:
    ```bash
    snakemake --profile workflow/profiles/slurm
    ```

!!! tip "report.html"
    After a complete run, `report.html` at the project root provides a self-contained
    interactive overview of all pipeline outputs — the recommended starting point for
    inspection.

## Rule Summary

| Rule | Inputs | Key Outputs | Resources |
|------|--------|-------------|-----------|
| `fit_curves` | `data/nadph.csv` | `ec50_fits.csv` | CPU |
| `hit_calling` | `ec50_fits.csv` | `cetsa_hits_ranked.csv` | CPU |
| `annotate` | fits + hits | `protein_sequences.fasta` | CPU |
| `system_analysis` | fits + hits + annotations | PCA, clustering, enrichment CSVs | CPU |
| `train_model` | fits + FASTA + cluster labels | `nadph_seq_head.pt`, embedding cache | **GPU** |
| `predict` | FASTA + checkpoint | `predictions_nadph_seq.csv` + saliency | **GPU** |
| `visualize` | predictions + fits + history | 14 evaluation plots | CPU |
| `network_analysis` | raw CSV | `network_graph.gexf`, modules | CPU |
| `curve_ml` | raw CSV | curve clusters + outliers | CPU |
| `bayesian_fit` | raw CSV + hits | Bayesian EC50 posteriors | CPU |
| `detailed_plots` | raw CSV + fits + hits | per-protein curves + GoF plot | CPU |
| `model_performance` | supervised + predictions | `model_performance_report.txt` | CPU |
