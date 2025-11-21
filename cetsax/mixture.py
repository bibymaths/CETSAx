"""
mixture.py
----------

Mixture modelling for CETSA NADPH response populations.
This module provides functions to build a standardized feature matrix
from CETSA sensitivity scores and redox axes, fit Gaussian Mixture Models (GMMs
) with BIC-based model selection, assign cluster labels and posterior
responsibilities, and optionally label clusters by sensitivity levels.
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

from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# 1. FEATURE MATRIX FOR MIXTURE MODELLING
# ------------------------------------------------------------

def build_mixture_features(
        sens_df: pd.DataFrame,
        redox_df: Optional[pd.DataFrame] = None,
        id_col: str = "id",
        feature_cols: Optional[List[str]] = None,
        include_redox_axes: bool = True,
        log_transform_ec50: bool = True,
) -> pd.DataFrame:
    """
    Build feature matrix for mixture modelling.

    Parameters
    ----------
    sens_df : DataFrame
        Typically output of sensitivity.compute_sensitivity_scores,
        with at least: id, EC50, delta_max, R2, NSS, ...

    redox_df : DataFrame, optional
        Output of redox.build_redox_axes, with columns:
            id, axis_direct, axis_indirect, axis_network, ...

    feature_cols : list of str or None
        Which columns from sens_df to use.
        If None, defaults to:
            ["EC50", "delta_max", "NSS", "R2"]

    include_redox_axes : bool
        If True and redox_df provided, add:
            ["axis_direct", "axis_indirect", "axis_network"]

    log_transform_ec50 : bool
        If True and "EC50" in features, replace with -log10(EC50)
        (so higher means stronger / more sensitive).

    Returns
    -------
    DataFrame
        id-indexed standardized feature matrix for mixture modelling.
    """
    if feature_cols is None:
        feature_cols = ["EC50", "delta_max", "NSS", "R2"]

    # Keep only available numeric columns
    cols = [c for c in feature_cols if c in sens_df.columns]
    if not cols:
        raise ValueError("No valid feature_cols found in sens_df.")

    feat = sens_df[[id_col] + cols].drop_duplicates(subset=[id_col]).set_index(id_col)

    # Optionally transform EC50 -> -log10(EC50)
    if log_transform_ec50 and "EC50" in feat.columns:
        ec50 = feat["EC50"].replace(0, np.nan)
        feat["minus_log10_EC50"] = -np.log10(ec50)
        feat = feat.drop(columns=["EC50"])

    # Attach redox axes if requested
    if include_redox_axes and redox_df is not None:
        red = redox_df[[id_col, "axis_direct", "axis_indirect", "axis_network"]].drop_duplicates(subset=[id_col])
        red = red.set_index(id_col)
        feat = feat.join(red, how="left")

    # Drop all-NaN columns
    feat = feat.dropna(axis=1, how="all")

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(feat.values.astype(float))
    feat_std = pd.DataFrame(X, index=feat.index, columns=feat.columns)

    return feat_std


# ------------------------------------------------------------
# 2. GMM WITH BIC-BASED MODEL SELECTION
# ------------------------------------------------------------

def fit_gmm_bic_grid(
        feat_df: pd.DataFrame,
        n_components_grid: List[int] = None,
        covariance_type: str = "full",
        random_state: int = 0,
) -> Dict[str, Any]:
    """
    Fit Gaussian Mixture Models for a grid of component numbers
    and select the best model via lowest BIC.

    Parameters
    ----------
    feat_df : DataFrame
        Standardized feature matrix indexed by id.

    n_components_grid : list of int or None
        List of component numbers to evaluate.
        If None, defaults to [1, 2, 3, 4, 5].

    covariance_type : {"full", "tied", "diag", "spherical"}
        Covariance structure for GMM.

    Returns
    -------
    dict with keys:
        - "best_model": fitted GaussianMixture
        - "best_k": number of components
        - "bic_table": DataFrame with BIC per k
    """
    if n_components_grid is None:
        n_components_grid = [1, 2, 3, 4, 5]

    X = feat_df.values
    bics = []
    models: Dict[int, GaussianMixture] = {}

    for k in n_components_grid:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        bics.append({"n_components": k, "BIC": bic})
        models[k] = gmm

    bic_df = pd.DataFrame(bics).sort_values("n_components")
    best_row = bic_df.loc[bic_df["BIC"].idxmin()]
    best_k = int(best_row["n_components"])
    best_model = models[best_k]

    return {
        "best_model": best_model,
        "best_k": best_k,
        "bic_table": bic_df,
    }


# ------------------------------------------------------------
# 3. CLUSTER ASSIGNMENT & SOFT RESPONSIBILITIES
# ------------------------------------------------------------

def assign_mixture_clusters(
        feat_df: pd.DataFrame,
        gmm: GaussianMixture,
        id_col: str = "id",
) -> pd.DataFrame:
    """
    Assign mixture clusters and posterior responsibilities for each protein.

    Parameters
    ----------
    feat_df : DataFrame
        Standardized feature matrix indexed by id.

    gmm : GaussianMixture
        Fitted GMM.

    Returns
    -------
    DataFrame
        id | cluster | resp_k... (one column per mixture component)
    """
    ids = feat_df.index.to_list()
    X = feat_df.values

    labels = gmm.predict(X)
    resp = gmm.predict_proba(X)

    n_components = gmm.n_components
    resp_cols = [f"resp_k{i}" for i in range(n_components)]

    out = pd.DataFrame(resp, index=ids, columns=resp_cols)
    out["cluster"] = labels
    out[id_col] = out.index

    # reorder columns: id, cluster, resp...
    cols = [id_col, "cluster"] + resp_cols
    return out[cols]


# ------------------------------------------------------------
# 4. OPTIONAL: SEMANTIC LABELING OF CLUSTERS
# ------------------------------------------------------------

def label_clusters_by_sensitivity(
        sens_df: pd.DataFrame,
        cluster_df: pd.DataFrame,
        id_col: str = "id",
        score_col: str = "NSS",
) -> pd.DataFrame:
    """
    Assign human-readable labels ("high", "medium", "low") to mixture clusters
    by ranking them by the mean of a chosen score (e.g. NSS, -log10 EC50).

    Parameters
    ----------
    sens_df : DataFrame
        Per-protein scores, must contain id_col and score_col.

    cluster_df : DataFrame
        Output of assign_mixture_clusters, must contain id_col and 'cluster'.

    Returns
    -------
    DataFrame
        cluster | mean_score | label
    """
    merged = pd.merge(
        cluster_df[[id_col, "cluster"]],
        sens_df[[id_col, score_col]],
        on=id_col,
        how="left",
    )

    grp = merged.groupby("cluster")[score_col].mean().reset_index()
    grp = grp.rename(columns={score_col: "mean_score"})

    # Rank clusters by mean score descending
    grp = grp.sort_values("mean_score", ascending=False).reset_index(drop=True)

    labels = ["high", "medium", "low"]
    if len(grp) > len(labels):
        # extend labels if more clusters than 3
        labels = labels + [f"cluster_{i}" for i in range(len(labels), len(grp))]

    grp["label"] = labels[: len(grp)]

    return grp
