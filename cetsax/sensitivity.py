"""
sensitivity.py
--------------

Proteome-wide NADPH sensitivity modelling.
This module provides functions to compute a unified NADPH Sensitivity Score (NSS)
per protein by integrating dose-response fit parameters, summarize sensitivity
at the pathway/module level, and quantify sensitivity heterogeneity across the proteome.

"""
# BSD 3-Clause License
#
# Copyright (c) 2025, Abhinav Mishra
# All rights reserved.
# Email: mishraabhinav36@gmail.com
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of Abhinav Mishra nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
