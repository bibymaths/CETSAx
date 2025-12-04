<img src="logo.svg" alt="CETSAx Logo" width="400"/>  

![uv](https://img.shields.io/badge/uv-env-orange)
![Snakemake](https://img.shields.io/badge/Snakemake-Workflow-blue?logo=snakemake)
![Python](https://img.shields.io/badge/Python-3.11.13-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?logo=pytorch) 

![ESM-2](https://img.shields.io/badge/ESM--2-Protein%20LM-green)
![License](https://img.shields.io/badge/License-BSD%203--Clause-lightgrey)

# **CETSAx–NADPH: Biological Insights & Explainable Protein–Metabolite Interaction Analysis**

This repository provides a streamlined workflow to uncover **how proteins respond to NADPH** using CETSA data.
It integrates **biophysical modeling**, **protein-language models**, and **explainable AI**, with minimal processing steps and fast execution via **Snakemake + uv**.

The focus is simple:

**→ quantify protein stability changes**
**→ explain why they happen**
**→ link sequence, domains, and motifs to NADPH-driven effects**

---

## 🌱 Biological Insight: What This Pipeline Actually Reveals

**1. Thermal stability as a metabolic sensor**
CETSA profiles how proteins shift in stability upon NADPH exposure. Curve fitting captures **EC50**, **direction**, and **strength** of stabilization/destabilization—directly mapping biochemical responsiveness.

**2. System-wide metabolic signatures**
Dimensionality reduction and clustering reveal groups of proteins with coordinated responses—often mapping to redox pathways, metabolic nodes, and structural complexes influenced by NADPH availability.

**3. Sequence-level determinants of NADPH responsiveness**
Using ESM-2, the pipeline learns sequence patterns linked to NADPH effects. With explainability (saliency, IG), the model highlights:

* catalytic/active-site residues
* redox-sensitive motifs
* cofactor-binding signatures
* structural “hotspots” that shift under NADPH

**This turns a black-box model into a biological hypothesis generator.**

**4. Network-level effects**
Co-stabilization networks expose modules: enzymes, scaffolds, or complexes that act together under a redox challenge.

---

## 🧠 Explainable AI: How the Pipeline Makes Predictions Interpretable

The ESM-based model isn’t just trained—it is made **biologically transparent**:

* **Saliency Maps** → residue-level signal of what the model used
* **Integrated Gradients** → robust attribution across the entire sequence
* **Global feature importance** → what amino acid patterns matter for NADPH response
* **Per-protein interpretability reports** → plots that map importance to domains, motifs, PTM hotspots

These explanations help you **link sequence → structure → function** in NADPH-driven stability changes.

---

## 🚀 Core Workflow

The Snakemake pipeline handles only the essential steps:

1. **Curve fitting** – clean EC50, slope, R²
2. **Hit calling** – confident responders vs non-responders
3. **Sequence annotation** – UniProt rescue for symbol/isoform mismatches
4. **System-level patterns** – PCA, clustering
5. **ESM-2 training** – sequence-based prediction of NADPH response
6. **Explainability** – saliency/IG attribution maps
7. **Biological integration** – network and domain-level interpretation

--- 

## 🚀 **Installation & Setup**

This project uses **uv** for fast, modern Python package management.

### **Clone the Repository**

```bash
git clone https://github.com/yourusername/cetsax-nadph.git
cd cetsax-nadph
```

> **Note:** A CUDA-capable GPU is recommended for efficient ESM-2 training.

## ⚙️ **Set Up the Environment**

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## 🏃 Run the Workflow

```bash
snakemake -c8
```

---

## 📊 Outputs

* **Protein-level EC50 curves**
* **Responder classifications**
* **Sequence attention heatmaps**
* **Integrated Gradients attribution**
* **Domain-aligned residue importance**
* **Cluster maps of global NADPH response**
* **Co-stabilization network modules**

These outputs directly support biological interpretation. 

---

## 📂 **Repository Structure**

```
├── config.yaml                     # Pipeline configuration
├── Snakefile                       # Workflow definition
├── pyproject.toml                  # Dependencies (uv-managed)
├── data/
│   └── nadph.csv                   # Input CETSA data
├── scripts/
│   ├── 01_fit_itdr_curves.py
│   ├── 02_hit_calling_and_diagnostics.py
│   ├── 03_system_level_analysis.py
│   ├── 04_seq_build_and_train.py
│   ├── 05_predict_nadph_effects.py
│   ├── 06_model_predict_results.py
│   ├── 07_network_analysis.py
│   ├── 08_curve_ml.py
│   ├── 09_bayesian_fit.py
│   └── 10_plot_curves.py
├── cetsax/                         # Core library code
└── results/
    ├── plots/
    ├── hit_calling/
    ├── system_analysis/
    ├── network/
    ├── curve_ml/
    ├── bayesian/
    └── detailed_plots/
```

---

## ⚙️ **Configuration**

All settings are controlled through `config.yaml`.

```yaml
input_csv: "data/nadph.csv"
python_bin: ".venv/bin/python"

# Deep Learning Hyperparameters
epochs: 10
batch_size: 8
task: "classification"   # classification or regression
device: "cuda"
```

---

## 🧠 **Model Architecture**

* **Backbone:** ESM-2 (t33_650M_UR50D)
* **Pooling:** Custom Attention Pooling
* **Head:** MLP with BatchNorm + Dropout
* **Loss:** Class-weighted Cross-Entropy (handles imbalance)

---

## 🔍 How to Interpret the Results

This section maps the main outputs of the pipeline to **biological meaning** and **model behaviour**.  
Use it as a checklist when you browse `results/`.

---

### 1. Curve Fits & EC50 Estimates

**Where to look**

* `results/hit_calling/`
* `results/bayesian/`
* `results/detailed_plots/` (per-protein curves)

**Key quantities**

* **EC50** – concentration at which the protein shows half-maximal stabilization/destabilization.
* **R² / fit quality** – reliability of the EC50 estimate.
* **Direction of effect**

  * Upward curve → stabilization (likely binding or complex formation).
  * Downward curve → destabilization (unfolding, competition, complex disruption).

**How to read it biologically**

* **Low EC50 + strong stabilization** → high apparent affinity / sensitivity to NADPH.
* **High EC50** → only responds at high NADPH; could be indirect or weak binding.
* **Poor R²** → treat the hit with caution; often noise, non-sigmoidal behaviour, or assay issues.

---

### 2. Hit Calling Tables

**Where to look**

* `results/hit_calling/` (summary tables + QC flags)

**What matters**

* Columns with hit labels like `strong`, `weak`, `non_responder`.
* QC metrics: minimum replicate count, fit convergence, R² thresholds.

**Biological interpretation**

* **Strong hits** – prime candidates for:

  * direct NADPH binding,
  * conformational stabilization via redox state,
  * membership in NADPH-dependent complexes.
* **Weak hits** – may still be interesting if they cluster in:

  * specific pathways,
  * known NADPH-related modules,
  * particular cellular compartments.

---

### 3. System-Level Patterns

**Where to look**

* `results/system_analysis/`

  * PCA projections
  * Cluster assignments
  * Pathway/GO summaries (if configured)

**How to read**

* **PCA plots** – show whether NADPH-responsive proteins cluster in specific “response modes” (e.g. redox enzymes, metabolic hubs).
* **Clusters** – each cluster is a candidate “response phenotype”:

  * early/strong stabilizers vs late/weak responders,
  * destabilized vs stabilized proteins,
  * pathway-specific modules.

Use these plots to move from **single proteins → coordinated systems**.

---

### 4. Sequence-Based Model Predictions

**Where to look**

* `results/plots/plot_1_confusion_matrix.png`
* `results/plots/plot_2_roc_curves.png`
* `results/plots/plot_5_ec50_correlation.png`
* `results/seq_predictions/` (if present: per-protein prediction tables)

**Key outputs**

* Predicted class or probability of being **NADPH-responsive**.
* Performance metrics:

  * **Confusion matrix** – how often the model is right/wrong.
  * **ROC / AUC** – how well it separates responders vs non-responders.
  * **Correlation with EC50** – whether the model’s confidence tracks biochemical strength.

**How to interpret**

* Good AUC and reasonable calibration → the model has captured **sequence features** of NADPH response.
* Misclassified proteins:

  * may signal noisy data or,
  * genuinely unusual biology (interesting edge cases).

---

### 5. Interpretability: Saliency & Integrated Gradients

**Where to look**

* `results/plots/plot_4_saliency_map.png`
* `results/plots/plot_8_residue_importance.png`
* Any per-protein attribution plots in `results/plots/` or `results/interpretability/`.

**What these show**

* **Saliency maps** – which residues the model relied on for its decision.
* **Integrated Gradients (IG)** – more stable attribution across the sequence.
* **Global importance plots** – amino acids or motif patterns that matter most overall.

**How to read them biologically**

For each protein:

1. Align high-importance residues with:

   * annotated domains (e.g. catalytic, regulatory),
   * cofactor-binding motifs,
   * known redox-sensitive cysteines or PTM sites.
2. Check whether highlighted residues:

   * sit in NAD(P)-binding folds,
   * overlap with known or predicted ligand pockets,
   * cluster in flexible loops that may shift with NADPH.

If the model focuses on **chemically sensible** residues (e.g. cofactor pocket, catalytic site), this supports a **mechanistically meaningful** signal rather than pure statistical artefact.

---

### 6. Network & Module-Level Effects

**Where to look**

* `results/network/`

  * co-stabilization networks
  * cluster/module annotations

**How to interpret**

* Nodes = proteins; edges = shared stability response patterns.
* Dense subgraphs / modules usually correspond to:

  * complexes,
  * pathways,
  * co-regulated redox systems.

Combine this with pathway / GO annotations to describe **NADPH-driven rewiring** of protein networks rather than isolated hits.

---

### 7. Sanity Checks & Common Pitfalls

When interpreting any result:

* **Check QC first** – poor fits or low coverage can create fake hits.
* **Beware single-protein stories** – prioritise patterns that repeat:

  * across replicates,
  * within pathways,
  * in multiple interpretability views (curve shape + sequence attribution + network context).
* **Use the model as a hypothesis generator**, not as proof:

  * high IG importance at a residue → candidate site for mutagenesis or follow-up experiments.

---

## 📚 **References**

### Data Sources

The dataset used in this repository originates from the **Cellular Thermal Shift Assay (CETSA)** study by Dziekan *et al.* I converted their provided `.Rdata` file into `.csv` format for use in this project.

**Reference:**
Dziekan, J.M., Wirjanata, G., Dai, L. *et al.*
*Cellular thermal shift assay for the identification of drug–target interactions in the Plasmodium falciparum proteome.*
**Nature Protocols**, 15, 1881–1921 (2020).
[DOI](https://doi.org/10.1038/s41596-020-0310-z)

**Original repository:**
[mineCETSA](https://github.com/nkdailingyun/mineCETSA)

**Data link:**
[RData](https://github.com/nkdailingyun/mineCETSA/tree/93ab6fc3c7186077f40a71c40300803aedd9f5ee/data)

---

### ESM Model

The ESM model used in this project (`esm2_t33_650M_UR50D`) comes from the following paper:

**Reference:**
Zeming Lin *et al.*
*Evolutionary-scale prediction of atomic-level protein structure with a language model.*
**Science**, 379, 1123–1130 (2023).
[DOI](https://doi.org/10.1126/science.ade2574)

**Model download:**
[Link](https://huggingface.co/facebook/esm2_t33_650M_UR50D)

---