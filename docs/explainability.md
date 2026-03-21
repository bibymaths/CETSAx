# Explainability

The sequence model is designed to be interpretable at the residue level.

## Methods

### Saliency

- Gradient-based importance
- Identifies residues influencing predictions

### Integrated Gradients

- Path-integrated attribution
- More stable and robust than raw gradients

## Outputs

- Per-residue importance scores
- Sequence-level importance maps
- Global amino acid importance summaries

## Interpretation

Important residues often align with:

- catalytic sites
- cofactor-binding regions
- redox-sensitive motifs
- structural hotspots

## Practical use

- Generate hypotheses for mutagenesis
- Identify potential binding regions
- Validate model behavior against known biology

Explainability bridges prediction and mechanism.