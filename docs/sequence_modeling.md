# Sequence Modeling

The sequence module predicts NADPH responsiveness directly from protein sequences.

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