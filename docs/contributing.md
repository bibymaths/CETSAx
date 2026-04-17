# Contributing

Contributions to CETSAx–NADPH are welcome, but they need to meet a clear standard. This project prioritizes correctness,
clarity, and biological relevance over feature volume.

---

## Philosophy

This is not a generic software project. It is a **scientific tool**.

Contributions should:

- improve correctness of the model or analysis
- increase interpretability
- enhance reproducibility
- extend functionality in a biologically meaningful way

Avoid adding features that increase complexity without clear scientific value.

---

## What You Can Contribute

### 1. Bug fixes

- Incorrect results or edge-case failures
- Numerical instability in fitting or scoring
- Data handling inconsistencies

---

### 2. Performance improvements

- Faster curve fitting
- Reduced memory usage in sequence models
- Better parallelization

---

### 3. New analysis modules

Examples:

- alternative scoring metrics
- improved network inference
- new clustering or latent methods

These should integrate cleanly into the existing pipeline.

---

### 4. Sequence modeling improvements

- better training strategies
- improved explainability methods
- integration of structural features

---

### 5. Documentation

- clearer explanations
- missing usage examples
- better interpretation guidance

---

## What Not to Contribute

- UI layers or dashboards without analytical value
- loosely tested experimental features
- large refactors without justification
- redundant implementations of existing functionality

---

## Development Setup

Clone the repository and install in editable mode:

```bash
git clone https://github.com/bibymaths/cetsax.git
cd cetsax-nadph

uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
pip install -e .
```

---

## Coding Guidelines

### General

* Keep functions small and focused
* Prefer explicit logic over abstraction
* Avoid hidden side effects

---

### Data handling

* Use `pandas.DataFrame` consistently
* Preserve column naming conventions (`id`, `condition`, metrics)
* Do not silently modify input data

---

### Numerical code

* Avoid unstable transformations
* Document assumptions (e.g. scaling, bounds)
* Prefer reproducible deterministic behavior

---

### Deep learning

* Keep training and inference clearly separated
* Avoid unnecessary GPU memory usage
* Document all hyperparameters

---

## Testing

Before submitting:

* Run the pipeline on a small dataset
* Verify outputs are consistent
* Check edge cases (missing data, low variance, etc.)

If possible:

* add unit tests
* include reproducible examples

---

## Submitting Changes

### 1. Create a branch

```bash id="p2x7oy"
git checkout -b feature/your-feature-name
```

---

### 2. Make focused commits

* One logical change per commit
* Clear commit messages

---

### 3. Open a Pull Request

Include:

* what the change does
* why it is needed
* how it was tested

If relevant, include before/after results.

---

## Review Process

Pull requests are evaluated based on:

* correctness
* clarity
* consistency with existing design
* biological relevance

Changes may be rejected if they:

* complicate the system unnecessarily
* introduce ambiguity in interpretation
* lack sufficient justification

---

## Style Expectations

* Write code as if it will be read, not just executed
* Avoid unnecessary cleverness
* Prefer clarity over brevity

---

## Communication

If you plan a larger change:

* open an issue first
* describe the idea clearly
* wait for feedback before implementing

This avoids wasted effort.

---

## Summary

Contribute if you can:

* make the model more accurate
* make the outputs more interpretable
* make the system more robust

Do not contribute just to add features.

The goal is a tool that produces results you would trust in a paper.
