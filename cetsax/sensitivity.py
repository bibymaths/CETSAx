"""
sensitivity.py
--------------

Proteome-wide NADPH sensitivity modelling.

Purpose:
    Build an integrated sensitivity score for each protein using
    fitted EC50, delta_max, Hill and R2 values and quantify
    proteome-wide heterogeneity in NADPH responsiveness.

This module supports:
    - Per-protein NADPH Sensitivity Score (NSS)
    - Pathway-level sensitivity summaries
    - Cross-protein sensitivity heterogeneity analysis
    - Replicate aggregation and confidence scoring
    - Feature matrices for latent-factor or multivariate models

NADPH Sensitivity Score (NSS):
    NSS = w1*(1/EC50_scaled)
        + w2*(delta_max_scaled)
        + w3*(Hill_scaled)
        + w4*(R2_scaled)

    where all components are min-max normalized or robustly scaled,
    then weighted and combined to capture affinity + effect size +
    curve quality.

Typical usage:

    df = pd.read_csv("ec50_fits.csv")
    sens = compute_sensitivity_scores(df)
    modules = summarize_sensitivity_by_pathway(sens, annotation_df)

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.preprocessing import RobustScaler


# ------------------------------------------------------------
# 1. NORMALIZATION HELPERS
# ------------------------------------------------------------

def _robust_scale(series: pd.Series) -> pd.Series:
    """Robust scaling using median + IQR; returns 0-1 clipped."""
    rs = RobustScaler()
    vals = series.values.reshape(-1, 1)
    try:
        scaled = rs.fit_transform(vals).flatten()
    except Exception:
        scaled = np.zeros_like(series.values)
    # squash to [0,1] using logistic transform
    scaled = 1 / (1 + np.exp(-scaled))
    return pd.Series(scaled, index=series.index)


def _inv_scale(series: pd.Series) -> pd.Series:
    """Invert scale for EC50 so low EC50 = high sensitivity."""
    # robust scale then invert
    s = _robust_scale(series)
    return 1.0 - s


# ------------------------------------------------------------
# 2. CORE SENSITIVITY SCORE
# ------------------------------------------------------------

def compute_sensitivity_scores(
        fits_df: pd.DataFrame,
        id_col: str = "id",
        cond_col: str = "condition",
        agg: str = "median",
        weights: Dict[str, float] = None,
) -> pd.DataFrame:
    """
    Compute a unified NADPH Sensitivity Score (NSS) per protein.

    Parameters
    ----------
    fits_df : DataFrame
        Output from fit_all_proteins, containing:
        [id, condition, EC50, delta_max, Hill, R2]

    agg : {"median", "mean"}
        How to aggregate across replicates.

    weights : dict or None
        Optional component weights:
            {"EC50": w1, "delta_max": w2, "Hill": w3, "R2": w4}
        If None, defaults to:
            EC50: 0.45, delta_max: 0.3, Hill: 0.15, R2: 0.10

    Returns
    -------
    DataFrame
        id | EC50 | delta_max | Hill | R2 | NSS | NSS_rank
    """
    if weights is None:
        weights = {
            "EC50": 0.45,
            "delta_max": 0.30,
            "Hill": 0.15,
            "R2": 0.10,
        }

    # Aggregate replicates
    if agg == "median":
        agg_df = (
            fits_df.groupby(id_col)
            .median(numeric_only=True)
            .reset_index()
        )
    elif agg == "mean":
        agg_df = (
            fits_df.groupby(id_col)
            .mean(numeric_only=True)
            .reset_index()
        )
    else:
        raise ValueError("agg must be 'median' or 'mean'")

    # Robust normalization
    ec50_scaled = _inv_scale(agg_df["EC50"])
    dm_scaled = _robust_scale(agg_df["delta_max"])
    h_scaled = _robust_scale(agg_df["Hill"])
    r2_scaled = _robust_scale(agg_df["R2"])

    # Weighted score
    NSS = (
            weights["EC50"] * ec50_scaled +
            weights["delta_max"] * dm_scaled +
            weights["Hill"] * h_scaled +
            weights["R2"] * r2_scaled
    )

    out = agg_df.copy()
    out["EC50_scaled"] = ec50_scaled
    out["delta_max_scaled"] = dm_scaled
    out["Hill_scaled"] = h_scaled
    out["R2_scaled"] = r2_scaled
    out["NSS"] = NSS
    out["NSS_rank"] = out["NSS"].rank(ascending=False)

    return out.sort_values("NSS", ascending=False)


# ------------------------------------------------------------
# 3. PATHWAY / MODULE SENSITIVITY SUMMARIES
# ------------------------------------------------------------

def summarize_sensitivity_by_pathway(
        sens_df: pd.DataFrame,
        annot_df: pd.DataFrame,
        id_col: str = "id",
        path_col: str = "pathway",
) -> pd.DataFrame:
    """
    Summarize NADPH sensitivity per pathway or complex.

    annot_df must map: id -> pathway/module

    Output: pathway | N | NSS_mean | NSS_top25 | EC50_median | delta_max_median
    """
    merged = pd.merge(sens_df, annot_df, on=id_col, how="left")

    summary = (
        merged.groupby(path_col)
        .agg(
            N=(id_col, "nunique"),
            NSS_mean=("NSS", "mean"),
            NSS_top25=("NSS", lambda x: x.quantile(0.75)),
            EC50_median=("EC50", "median"),
            delta_max_median=("delta_max", "median"),
        )
        .sort_values("NSS_mean", ascending=False)
    )

    return summary.reset_index()


# ------------------------------------------------------------
# 4. SENSITIVITY HETEROGENEITY ACROSS PROTEOME
# ------------------------------------------------------------

def compute_sensitivity_heterogeneity(
        sens_df: pd.DataFrame,
        bins: int = 50,
) -> Dict[str, Any]:
    """
    Quantify how spread-out sensitivity is across the proteome.

    Returns:
        dict with:
            - histogram
            - Gini coefficient (inequality)
            - top10% NSS threshold
    """
    NSS = sens_df["NSS"].values
    NSS_sorted = np.sort(NSS)

    # Gini coefficient for inequality of sensitivity
    idx = np.arange(1, NSS_sorted.size + 1)
    gini = (np.sum((2 * idx - NSS_sorted.size - 1) * NSS_sorted)) / (
            NSS_sorted.size * np.sum(NSS_sorted)
    )

    hist_vals, hist_edges = np.histogram(NSS, bins=bins)

    return {
        "gini": float(gini),
        "hist": hist_vals,
        "edges": hist_edges,
        "top10_threshold": float(np.quantile(NSS, 0.90)),
    }
