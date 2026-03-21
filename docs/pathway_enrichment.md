# Pathway Enrichment

Once protein-level sensitivity is computed, the system aggregates responses at the pathway or module level.

## Two complementary approaches

### 1. Effect size aggregation

For each pathway:

- Mean NSS
- Median NSS
- Upper quantile (top responders)
- Number of proteins

This captures how strongly a pathway responds overall.

### 2. Statistical enrichment

Two statistical frameworks are used:

- Fisher’s exact test for binary hit sets
- Mann–Whitney U test for continuous scores

Both are corrected using Benjamini–Hochberg FDR.

## Why both methods

- Binary enrichment identifies strong responders
- Continuous enrichment detects subtle but consistent shifts

Using both provides a more complete view.

## Interpretation

- Enriched pathways indicate coordinated NADPH sensitivity
- High NSS pathways often correspond to:
    - redox systems
    - metabolic hubs
    - cofactor-dependent enzymes

This step moves analysis from individual proteins to biological systems.