"""
latent.py
---------

Latent-factor modelling of CETSA NADPH responsiveness.
This module provides functions to build a feature matrix from
CETSA sensitivity scores and redox axes, and to perform
dimensionality reduction using PCA and Factor Analysis (FA).
It also includes utilities to merge latent representations
back to protein metadata tables.

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

import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# 1. FEATURE MATRIX CONSTRUCTION
# ------------------------------------------------------------

def build_feature_matrix(
        sens_df: pd.DataFrame,
        redox_df: Optional[pd.DataFrame] = None,
        id_col: str = "id",
        base_features: Optional[List[str]] = None,
        include_redox_axes: bool = True,
) -> pd.DataFrame:
    """
    Build a standardized feature matrix per protein.

    Parameters
    ----------
    sens_df : DataFrame
        Output of sensitivity.compute_sensitivity_scores, expected columns:
            id, EC50, delta_max, Hill, R2, NSS, EC50_scaled, ...

    redox_df : DataFrame, optional
        Output of redox.build_redox_axes, expected columns:
            id, axis_direct, axis_indirect, axis_network, redox_role, ...

    base_features : list of str or None
        Which numeric columns from sens_df to include as features.
        If None, defaults to:
            ["EC50", "delta_max", "Hill", "R2", "NSS",
             "EC50_scaled", "delta_max_scaled", "Hill_scaled", "R2_scaled"]

    include_redox_axes : bool
        If True and redox_df is provided, will also include:
            ["axis_direct", "axis_indirect", "axis_network"]
        as features.
    id_col : str
        Column name for protein identifier in sens_df and redox_df.

    Returns
    -------
    DataFrame
        Feature matrix indexed by id:
            index: id
            columns: selected features (standardized)
    """
    if base_features is None:
        base_features = [
            "EC50",
            "delta_max",
            "Hill",
            "R2",
            "NSS",
            "EC50_scaled",
            "delta_max_scaled",
            "Hill_scaled",
            "R2_scaled",
        ]

    # Keep only numeric base features that exist
    available = [f for f in base_features if f in sens_df.columns]
    if not available:
        raise ValueError("No valid base_features found in sens_df.")

    feat = sens_df[[id_col] + available].drop_duplicates(subset=[id_col]).set_index(id_col)

    # Attach redox axes if requested
    if include_redox_axes and redox_df is not None:
        red = redox_df[[id_col, "axis_direct", "axis_indirect", "axis_network"]].drop_duplicates(subset=[id_col])
        red = red.set_index(id_col)
        feat = feat.join(red, how="left")

    # Drop columns that are entirely NaN
    feat = feat.dropna(axis=1, how="all")

    # Standardize features (mean=0, std=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(feat.values.astype(float))
    feat_std = pd.DataFrame(X, index=feat.index, columns=feat.columns)

    return feat_std


# ------------------------------------------------------------
# 2. PCA
# ------------------------------------------------------------

def fit_pca(
        feat_df: pd.DataFrame,
        n_components: int = 3,
) -> Dict[str, Any]:
    """
    Fit PCA on the standardized feature matrix.

    Parameters
    ----------
    feat_df : DataFrame
        Feature matrix as returned by build_feature_matrix, indexed by id.

    n_components : int
        Number of principal components to compute.

    Returns
    -------
    dict with keys:
        - "model": fitted sklearn PCA object
        - "scores": DataFrame with PC coordinates per id
        - "loadings": DataFrame with feature loadings per component
        - "explained_variance_ratio": array of variance fractions
    """
    ids = feat_df.index.to_list()
    X = feat_df.values

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    score_cols = [f"PC{i + 1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, index=ids, columns=score_cols)

    loadings = pca.components_.T
    load_cols = score_cols
    loadings_df = pd.DataFrame(loadings, index=feat_df.columns, columns=load_cols)

    return {
        "model": pca,
        "scores": scores_df,
        "loadings": loadings_df,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


# ------------------------------------------------------------
# 3. FACTOR ANALYSIS
# ------------------------------------------------------------

def fit_factor_analysis(
        feat_df: pd.DataFrame,
        n_components: int = 3,
) -> Dict[str, Any]:
    """
    Fit Factor Analysis (FA) on the standardized feature matrix.

    FA often yields more interpretable latent factors than PCA
    when features are noisy and partially redundant.

    Parameters
    ----------
    feat_df : DataFrame
        Feature matrix as returned by build_feature_matrix, indexed by id.

    n_components : int
        Number of latent factors.

    Returns
    -------
    dict with keys:
        - "model": fitted sklearn FactorAnalysis object
        - "scores": DataFrame with factor scores per id
        - "loadings": DataFrame with feature loadings per factor
    """
    ids = feat_df.index.to_list()
    X = feat_df.values

    fa = FactorAnalysis(n_components=n_components)
    scores = fa.fit_transform(X)

    score_cols = [f"F{i + 1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, index=ids, columns=score_cols)

    loadings = fa.components_.T
    load_cols = score_cols
    loadings_df = pd.DataFrame(loadings, index=feat_df.columns, columns=load_cols)

    return {
        "model": fa,
        "scores": scores_df,
        "loadings": loadings_df,
    }


# ------------------------------------------------------------
# 4. ATTACH LATENT COORDS BACK TO METADATA
# ------------------------------------------------------------

def attach_latent_to_metadata(
        meta_df: pd.DataFrame,
        latent_df: pd.DataFrame,
        id_col: str = "id",
) -> pd.DataFrame:
    """
    Merge latent coordinates back to a per-protein metadata table.

    Parameters
    ----------
    meta_df : DataFrame
        Any per-protein table with column id_col (e.g. sens_df, redox_df).

    latent_df : DataFrame
        Latent representation indexed by id (e.g. PCA scores or FA scores).

    id_col : str
        Column name for protein identifier in meta_df and latent_df.

    Returns
    -------
    DataFrame
        meta_df with latent columns appended.
    """
    latent_df = latent_df.copy()
    latent_df[id_col] = latent_df.index
    merged = pd.merge(meta_df, latent_df, on=id_col, how="left")
    return merged
