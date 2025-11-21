"""
ml.py
-----
Machine learning utilities for CETSA dose-response curve analysis.
This module provides functions to extract features from dose-response curves
using PCA, classify curves using KMeans clustering, and detect outlier curves
based on z-score heuristics.
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

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def extract_curve_features(df: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
    """
    Reduce dose-response curves to principal components.
    Parameters
    ----------
    df : DataFrame
        Data Frame containing dose-response data with 'id' column and dose columns.
    n_components : int
        Number of PCA components to extract.
    Returns
    -------
    DataFrame
        Data Frame with PCA features per protein ID.
    """
    dose_mat = df.groupby("id")[df.columns[df.columns.str.contains("e-")]].mean()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dose_mat)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    feat_df = pd.DataFrame(
        X_pca,
        index=dose_mat.index,
        columns=[f"PC{i + 1}" for i in range(n_components)]
    )
    return feat_df


def classify_curves_kmeans(
        features: pd.DataFrame,
        k: int = 4
) -> pd.DataFrame:
    """
    Apply KMeans to curve embeddings (e.g. PCA features).
    Parameters
    ----------
    features : DataFrame
        Feature matrix per protein (e.g. output of extract_curve_features).
    k : int
        Number of clusters.
    Returns
    -------
    DataFrame
        Input features with an additional 'cluster' column.
    """
    km = KMeans(n_clusters=k, n_init="auto")
    labels = km.fit_predict(features)

    out = features.copy()
    out["cluster"] = labels
    return out


def detect_outliers(features: pd.DataFrame) -> pd.DataFrame:
    """
    Simple z-score heuristic for outlier curves.

    #TODO : Maybe Replace with robust Mahalanobis distance or isolation forest.

    Parameters
    ----------
    features : DataFrame
        Feature matrix per protein (e.g. output of extract_curve_features).
    Returns
    -------
    DataFrame
        Data Frame with boolean 'outlier' column.
    """
    z = np.abs((features - features.mean()) / features.std())
    out = (z > 3).any(axis=1)

    result = pd.DataFrame({
        "outlier": out.astype(bool)
    }, index=features.index)

    return result
