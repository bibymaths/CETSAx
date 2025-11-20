"""
bayes.py
--------

Purpose:
    Implement hierarchical Bayesian modelling of CETSA EC50 parameters.

    This supports:
    - Joint inference of EC50 across replicates (per protein)
    - Hierarchical pooling across pathways/modules
    - Posterior distributions for EC50, Hill, Emax
    - Bayesian uncertainty quantification for hit calling
    - Robustness against noisy replicates and weak signals

    Requires PyMC or NumPyro (not mandatory in base install).
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, Any
import pymc as pm
import arviz as az

def bayesian_fit_ec50(df: pd.DataFrame, protein_id: str) -> Dict[str, Any]:
    """
    Fit a hierarchical Bayesian EC50 model for a single protein
    across replicates.

    Returns:
        dict containing PyMC model, posterior samples, and summary.
    """
    if pm is None:
        raise ImportError("PyMC is required for Bayesian inference.")

    subset = df[df["id"] == protein_id]
    if subset.empty:
        raise ValueError(f"No data for protein {protein_id}")

    doses = subset.columns[subset.columns.str.contains("e-")].astype(float)
    y = subset[doses].values  # (replicates x doses)

    with pm.Model() as model:
        # Priors
        logEC50 = pm.Normal("logEC50", mu=-3, sigma=2)
        Hill = pm.HalfNormal("Hill", sigma=2)
        E0 = pm.Normal("E0", mu=1, sigma=0.1)
        Emax = pm.Normal("Emax", mu=1, sigma=0.5)

        EC50 = pm.Deterministic("EC50", 10**logEC50)

        # Likelihood
        def itdr(c, E0, Emax, logEC50, Hill):
            return E0 + (Emax - E0) / (1 + (10**logEC50 / c)**Hill)

        mu = itdr(doses.values, E0, Emax, logEC50, Hill)
        sigma = pm.HalfNormal("sigma", sigma=0.05)

        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(1000, tune=1000, chains=2, target_accept=0.9)

    return {
        "model": model,
        "trace": trace,
        "summary": az.summary(trace),
    }
