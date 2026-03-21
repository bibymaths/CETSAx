# Latent Analysis

The latent module reduces high-dimensional protein features into interpretable axes.

## Feature construction

Features include:

- EC50, Δmax, Hill, R²
- NSS
- Scaled versions of all parameters
- Optional redox-related features

All features are standardized before analysis.

## Methods

### Principal Component Analysis (PCA)

- Captures directions of maximum variance
- Useful for visualization and clustering

### Factor Analysis (FA)

- Extracts latent factors
- Often more interpretable when features are noisy

## Outputs

- Latent coordinates per protein
- Loadings for each feature
- Explained variance (for PCA)

## Interpretation

- Latent axes represent underlying biological trends
- Loadings reveal which parameters drive variation
- Clusters in latent space suggest distinct response modes

This step compresses complex behavior into interpretable structure.