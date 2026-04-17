# CETSAx–NADPH

<img src="assets/logo.svg" alt="CETSAx Logo" width="300"/>  

![uv](https://img.shields.io/badge/uv-env-orange) 
![conda](https://img.shields.io/badge/Conda-Package%20Manager-green?logo=anaconda)
![Snakemake](https://img.shields.io/badge/Snakemake-Workflow-blue?logo=snakemake)
![Python](https://img.shields.io/badge/Python-3.11.13-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?logo=pytorch) 

![ESM-2](https://img.shields.io/badge/ESM--2-Protein%20LM-green)
![License](https://img.shields.io/badge/License-BSD%203--Clause-lightgrey)

CETSAx–NADPH is a computational framework for analyzing protein stability responses to NADPH using CETSA data. It
combines biophysical modeling, statistical scoring, systems-level analysis, and sequence-based learning into a single
pipeline.

At its core, the system answers three questions:

- Which proteins respond to NADPH?
- How strong and reliable is that response?
- Why do these responses occur at the molecular and sequence level?

The pipeline moves from raw dose–response measurements to interpretable biological insights across multiple scales.

## What makes this system different

- It prioritizes **biophysical interpretability**, not just statistical fit.
- It integrates **protein-level, pathway-level, and sequence-level signals**.
- It includes **explainable AI**, allowing residue-level interpretation of predictions.

## High-level workflow

- Curve fitting of CETSA dose–response data
- Sensitivity scoring across the proteome
- Pathway and network-level aggregation
- Latent structure discovery
- Sequence-based modeling and interpretation

Each component is modular but designed to work coherently as a system.

## Intended audience

This framework is designed for:

- Computational biologists working with CETSA or thermal profiling
- Systems biologists studying metabolic or redox responses
- Researchers interested in interpretable protein language models