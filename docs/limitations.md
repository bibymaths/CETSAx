# Limitations

This section outlines the conceptual, statistical, and computational limitations of CETSAx–NADPH. These are not
implementation flaws; they reflect the assumptions and constraints of the underlying methodology.

Understanding these limitations is necessary for correct interpretation of results.

---

## 1. Model Assumptions

### Logistic response constraint

The dose–response model assumes a sigmoidal relationship between concentration and stability.

- Proteins with non-monotonic or multi-phase behavior are not well captured.
- Complex binding mechanisms (e.g. multi-site, allosteric effects) are reduced to a single effective EC50.

As a result, some biologically valid responses may be filtered out or mischaracterized.

---

### Monotonic smoothing

Isotonic regression enforces monotonicity.

- This removes noise but also removes genuine non-monotonic structure.
- Subtle transitions or intermediate states may be flattened.

---

## 2. Sensitivity Scoring

### Heuristic weighting

The NADPH Sensitivity Score (NSS) is a weighted combination of features.

- Weights are biologically motivated but not learned from data.
- Different weighting schemes may change protein rankings.

The score should therefore be treated as a **relative ranking**, not an absolute quantity.

---

### Dependence on fit quality

Sensitivity depends on fitted parameters.

- Poor fits propagate into NSS.
- Filtering reduces noise but may remove borderline cases.

---

## 3. Data Limitations

### CETSA-specific constraints

CETSA measures thermal stability, not binding directly.

- Stabilization may reflect:
    - ligand binding
    - complex formation
    - indirect metabolic effects

- Destabilization may arise from:
    - competition
    - conformational shifts
    - degradation or aggregation

Therefore, CETSA signals are **context-dependent proxies**, not direct mechanistic measurements.

---

### Replicate variability

- Limited replicates reduce reliability of EC50 estimation.
- High noise at low concentrations can bias fits.

---

### Coverage bias

- Not all proteins are equally detected.
- Abundant or stable proteins are overrepresented.

This introduces systematic bias in downstream analyses.

---

## 4. Pathway and Network Analysis

### Correlation-based networks

Co-stabilization networks are based on correlation.

- Correlation does not imply causation.
- Shared patterns may arise from indirect effects or experimental artifacts.

---

### Annotation dependence

Pathway enrichment depends on annotation quality.

- Missing or incorrect annotations affect results.
- Broad pathway definitions dilute specificity.

---

## 5. Latent Representation

### Linear assumptions (PCA)

- PCA captures only linear structure.
- Nonlinear relationships may be missed.

---

### Factor interpretation

- Latent factors are not uniquely identifiable.
- Biological interpretation depends on feature loadings and context.

---

## 6. Sequence Modeling

### Data size constraints

- Performance depends on the number and quality of labeled proteins.
- Small datasets limit generalization.

---

### Label noise

Labels are derived from experimental fits.

- Errors in EC50 or classification propagate into training.
- The model may learn dataset-specific artifacts.

---

### Model bias

- ESM embeddings encode evolutionary information, not experimental context.
- Predictions reflect both sequence signal and pretrained biases.

---

### Limited structural awareness

- The model operates on sequence, not explicit 3D structure.
- Structural context is inferred implicitly, not modeled directly.

---

## 7. Explainability

### Gradient-based attribution limits

- Saliency and Integrated Gradients depend on model gradients.
- These methods highlight sensitivity, not causality.

High importance does not guarantee biological relevance.

---

### Resolution vs interpretation

- Residue-level signals can be noisy.
- Aggregation across proteins is required for robust conclusions.

---

## 8. Computational Constraints

### GPU requirements

- Sequence modeling is resource-intensive.
- Limited GPU memory restricts batch size and model complexity.

---

### Runtime scaling

- Curve fitting scales with number of proteins × conditions.
- Large datasets increase runtime significantly.

---

## 9. Interpretation Risks

### Overinterpretation of single proteins

- Individual hits can be misleading.
- Reliable conclusions require consistent patterns across:
    - replicates
    - pathways
    - multiple analysis layers

---

### Black-box misuse

- The sequence model can produce confident predictions even when data is weak.
- Interpretability tools mitigate this but do not eliminate the risk.

---

## 10. Summary

CETSAx–NADPH provides a structured and interpretable framework, but it operates under:

- simplified biophysical assumptions
- noisy experimental inputs
- statistical and computational constraints

Results should be interpreted as:

- **hypothesis-generating**, not definitive proof
- strongest when supported across multiple layers of analysis

Careful validation with orthogonal experiments is recommended.