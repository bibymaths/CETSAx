<img src="assets/logo.svg" alt="CETSAx Logo" width="800"/>  

![uv](https://img.shields.io/badge/uv-env-orange) 
![conda](https://img.shields.io/badge/Conda-Package%20Manager-green?logo=anaconda)
![Snakemake](https://img.shields.io/badge/Snakemake-Workflow-blue?logo=snakemake)
![Python](https://img.shields.io/badge/Python-3.11.13-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?logo=pytorch) 

![ESM-2](https://img.shields.io/badge/ESM--2-Protein%20LM-green)
![License](https://img.shields.io/badge/License-BSD%203--Clause-lightgrey)

# CETSAx–NADPH

CETSAx–NADPH is a pipeline for analyzing protein stability responses to NADPH using CETSA data.  
It combines dose–response modeling, system-level analysis, and sequence-based learning into a single workflow.

---

## Documentation

Full documentation:

https://bibymaths.github.io/CETSAx/

Key sections:

- Overview: https://bibymaths.github.io/CETSAx/
- Usage: https://bibymaths.github.io/CETSAx/usage/
- Scientific model (curve fitting): https://bibymaths.github.io/CETSAx/curve_fitting/
- Sensitivity scoring: https://bibymaths.github.io/CETSAx/sensitivity_scoring/
- API reference: https://bibymaths.github.io/CETSAx/api/
- Limitations: https://bibymaths.github.io/CETSAx/limitations/

---

## What it does

- Fits CETSA dose–response curves (EC50, slope, R²)
- Identifies responsive proteins (hit calling)
- Extracts system-level patterns (PCA, clustering, networks)
- Learns sequence determinants of response (ESM-2)
- Provides residue-level interpretability (saliency, integrated gradients)

This is designed as a hypothesis-generation framework rather than a standalone fitting tool.

---

## Quickstart

Clone the repository:

```bash
git clone https://github.com/bibymaths/CETSAx.git
cd CETSAx
````

Set up environment (recommended):

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Run the pipeline:

```bash
snakemake --cores 8
```

---

## Inputs

* CETSA dose–response dataset (`.csv`)
* Configuration via `config.yaml`

Details:

[https://bibymaths.github.io/CETSAx/usage/](https://bibymaths.github.io/CETSAx/usage/)

---

## Outputs

* Curve fits (EC50, Δmax, R²)
* Hit classifications
* System-level analyses (PCA, clusters)
* Sequence model predictions
* Interpretability maps
* Co-stabilization networks

Interpretation guide:

[https://bibymaths.github.io/CETSAx/usage/#how-to-work-with-results](https://bibymaths.github.io/CETSAx/usage/#how-to-work-with-results)

---

## Project Structure

```
cetsax/        # core library
scripts/       # pipeline steps
data/          # input
results/       # outputs
Snakefile      # workflow
config.yaml    # configuration
```

---

## Contributing

[https://bibymaths.github.io/CETSAx/contributing/](https://bibymaths.github.io/CETSAx/contributing/)

---

## Citation

[https://bibymaths.github.io/CETSAx/citation/](https://bibymaths.github.io/CETSAx/citation/)

---

## License

BSD 3-Clause