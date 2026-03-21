# Sensitivity Scoring

CETSA responses cannot be summarized by a single parameter. The pipeline therefore computes a composite score that
captures multiple aspects of protein behavior.

## NADPH Sensitivity Score (NSS)

The NSS integrates:

- EC50 (inverted, lower is stronger)
- Δmax (magnitude of response)
- Hill coefficient (shape of response)
- R² (fit reliability)

Each component is:

- Robustly scaled using median and IQR
- Transformed to a bounded range
- Combined using weighted aggregation

## Default weighting

- EC50: dominant contribution
- Δmax: secondary importance
- Hill: moderate influence
- R²: reliability adjustment

This reflects biological priorities, where affinity-like behavior (EC50) is most informative.

## Replicate handling

- Replicates are aggregated per protein using median or mean.
- This reduces noise and stabilizes downstream analysis.

## Output

- NSS (continuous score per protein)
- NSS rank across the proteome
- Scaled versions of all contributing features

## Interpretation

- High NSS → strong and reliable NADPH response
- Low NSS → weak or no response
- Distribution of NSS reflects system-wide sensitivity patterns

The NSS provides a unified, interpretable measure of responsiveness.