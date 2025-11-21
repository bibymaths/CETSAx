# **CETSAx-NADPH: Deep Learning & Systems Analysis of Protein-Metabolite Interactions**
 
![Snakemake](https://img.shields.io/badge/Snakemake-Workflow-blue?logo=snakemake)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![uv](https://img.shields.io/badge/uv-fast%20package%20manager-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![ESM-2](https://img.shields.io/badge/ESM--2-Protein%20LM-green)
![License](https://img.shields.io/badge/License-BSD%203--Clause-lightgrey)

CETSAx is a comprehensive bioinformatics and machine learning pipeline for analyzing Cellular Thermal Shift Assay (CETSA) data. It combines classical biophysical curve fitting with modern Protein Language Models (ESM-2) to characterize and predict protein responsiveness to **NADPH**.

This repository provides an automated **Snakemake** workflow that processes raw proteomics data, fits ITDR curves, annotates proteins, fine-tunes a deep learning model, and visualizes binding sites through interpretability methods.

---

## 🧬 **Key Features**

* **Automated Curve Fitting**
  Robust EC50 and R² estimation for thousands of proteins in multiplexed proteomics experiments.

* **Smart Annotation**
  Retrieves gene symbols and sequences from UniProt/MyGene, with isoform rescue logic for mismatching proteins.

* **System-Level Analysis**
  PCA, GMM clustering, and pathway enrichment for understanding global NADPH effects.

* **Deep Learning (ESM-2)**
  Fine-tunes the 650M-parameter ESM-2 model to predict NADPH responsiveness from sequence alone.

* **Interpretability**
  Saliency Maps and Integrated Gradients to identify crucial amino acids driving binding.

* **Network Analysis**
  Constructs co-stabilization networks to detect responsive protein modules.

* **Unsupervised Curve ML**
  PCA + K-Means on curve shapes to identify phenotypes and outliers.

* **Bayesian Validation**
  MCMC sampling for posterior EC50 estimation.

* **Reproducible Workflow**
  Entire pipeline orchestrated using **Snakemake** with **uv** for strict dependency management.

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

## 🚀 **Installation & Setup**

This project uses **uv** for fast, modern Python package management.

### **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/cetsax-nadph.git
cd cetsax-nadph
```

### **2. Set up the Environment**

```bash
# Create environment
uv venv

# Activate
source .venv/bin/activate

# Install dependencies
uv pip install torch pandas numpy scipy matplotlib seaborn scikit-learn fair-esm mygene snakemake pymc arviz networkx
```

> **Note:** A CUDA-capable GPU is recommended for efficient ESM-2 training.

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

## 🏃‍♂️ **Running the Pipeline**

Run with up to 8 CPU cores:

```bash
snakemake -c8 --use-conda
```

### **Pipeline Steps (DAG Overview)**

1. **Fit Curves** – Sigmoid ITDR fitting
2. **Hit Calling** – QC, filtering, ranking
3. **Annotate** – UniProt and sequence mapping
4. **System Analysis** – PCA, GMM
5. **Train** – Fine-tuning ESM-2
6. **Predict** – Inference + Integrated Gradients
7. **Visualize** – Diagnostics and performance plots
8. **Network Analysis** – Co-stabilization modules
9. **Curve ML** – Shape clustering and outlier detection
10. **Bayesian Fit** – MCMC EC50 estimation
11. **Detailed Plots** – Per-protein curve visualization

---

## 📊 **Outputs & Visualization**

Generated under `results/plots/`.

| File                            | Description                                |
| ------------------------------- | ------------------------------------------ |
| `plot_1_confusion_matrix.png`   | Accuracy across strong/medium/weak binders |
| `plot_2_roc_curves.png`         | AUC metrics                                |
| `plot_4_saliency_map.png`       | Residue-level feature importance           |
| `plot_5_ec50_correlation.png`   | Correlation of model confidence vs. EC50   |
| `plot_8_residue_importance.png` | Global amino acid importance               |

### Example Saliency Map

*Generated after the ESM-2 model is trained.*

---

## 🧠 **Model Architecture**

* **Backbone:** ESM-2 (t33_650M_UR50D)
* **Pooling:** Custom Attention Pooling
* **Head:** MLP with BatchNorm + Dropout
* **Loss:** Class-weighted Cross-Entropy (handles imbalance)

---

## 📚 **References**

* Lin et al. *Evolutionary-scale prediction of atomic-level protein structure with a language model.*

---