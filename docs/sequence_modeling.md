# Sequence Modeling

The sequence module predicts NADPH responsiveness directly from protein sequences.

!!! info "ESM2 Model Flexibility"
    Any ESM2 protein model from [Facebook/Meta on HuggingFace](https://huggingface.co/facebook)
    can be used (e.g., `facebook/esm2_t6_8M_UR50D` through `facebook/esm2_t48_15B_UR50D`).

    The most basic model (`esm2_t6_8M_UR50D`) was used during development due to
    GPU resource constraints. Testing was performed on a **Tesla T4 (16 GB VRAM)**.
    Larger models will require more VRAM — scale your model selection accordingly
    via `config.yaml`.

## Backbone

- ESM-2 protein language model
- Pretrained on large protein sequence corpora

## Training modes

- Pooled embeddings (fast)
- Residue representations (interpretability)
- Token-level training (full fine-tuning)

## Architecture

- Attention pooling over residues
- Fully connected prediction head
- Supports classification and regression

## Training features

- Focal loss for class imbalance
- Gradient accumulation for memory efficiency
- Mixed precision for speed

## Labels

Derived from experimental data:

- Classification: strong vs weak responders
- Regression: transformed EC50 or NSS

## Interpretation

The model learns sequence patterns associated with NADPH sensitivity, enabling prediction for unseen proteins.

This connects sequence information to biochemical behavior.