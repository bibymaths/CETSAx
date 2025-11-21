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


def bayesian_fit_ec50(
    df: pd.DataFrame,
    protein_id: str,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 1,
    progressbar: bool = True
) -> Dict[str, Any]:
    """
    Fit a hierarchical Bayesian EC50 model for a single protein
    across replicates.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing CETSA data.
    protein_id : str
        ID of the protein to fit.
    draws : int
        Number of samples to draw from the posterior (per chain).
    tune : int
        Number of tuning steps (burn-in).
    chains : int
        Number of independent MCMC chains to run.
    cores : int
        Number of CPU cores to use for parallel sampling.
        Set to 1 if running this function inside a joblib loop.
        Set to equal 'chains' if running on a single protein.
    progressbar : bool
        Whether to show the PyMC progress bar.

    Returns
    -------
    dict containing PyMC model, posterior samples (trace), and summary.
    """
    if pm is None:
        raise ImportError("PyMC is required for Bayesian inference.")

    subset = df[df["id"] == protein_id]
    if subset.empty:
        raise ValueError(f"No data for protein {protein_id}")

    # 1. Identify dose columns (strings)
    dose_cols = subset.columns[subset.columns.str.contains("e-")]

    # 2. Get numerical values for the model (floats)
    # We need these for the 'itdr' mathematical function
    doses_val = dose_cols.astype(float)

    # 3. Get intensity data using string column names
    y = subset[dose_cols].values  # (replicates x doses)

    with pm.Model() as model:
        # Priors
        logEC50 = pm.Normal("logEC50", mu=-3, sigma=2)
        Hill = pm.HalfNormal("Hill", sigma=2)
        E0 = pm.Normal("E0", mu=1, sigma=0.1)
        Emax = pm.Normal("Emax", mu=1, sigma=0.5)

        EC50 = pm.Deterministic("EC50", 10 ** logEC50)

        # Likelihood
        def itdr(c, E0, Emax, logEC50, Hill):
            return E0 + (Emax - E0) / (1 + (10 ** logEC50 / c) ** Hill)

        # Use numerical doses here
        mu = itdr(doses_val.values, E0, Emax, logEC50, Hill)
        sigma = pm.HalfNormal("sigma", sigma=0.05)

        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        # Parallel sampling handled here via 'cores' argument
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=0.9,
            progressbar=progressbar
        )

    return {
        "model": model,
        "trace": trace,
        "summary": az.summary(trace),
    }