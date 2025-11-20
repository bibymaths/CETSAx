"""
ml.py
-----

Purpose:
    Implement machine learning methods for CETSA curve classification.

    This module enables:
    - Unsupervised clustering (KMeans, HDBSCAN, spectral)
    - Curve shape embedding via PCA, UMAP, or autoencoders
    - Supervised classification of curve types (if labels available)
    - Identification of atypical or noisy curves
    - Feature extraction from dose-response signatures

    Useful for:
    - Distinguishing direct vs indirect stabilizers
    - Identifying noisy/flat/unreliable curves
    - Extracting high-level curve phenotypes
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def extract_curve_features(df: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
    """
    Reduce dose-response curves to principal components.

    Returns:
        DataFrame with PC1–PCn features per protein.
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

    Returns:
        DataFrame with cluster labels.
    """
    km = KMeans(n_clusters=k, n_init="auto")
    labels = km.fit_predict(features)

    out = features.copy()
    out["cluster"] = labels
    return out


def detect_outliers(features: pd.DataFrame) -> pd.DataFrame:
    """
    Simple z-score heuristic for outlier curves.
    Can be replaced with isolation forest or HDBSCAN later.
    """
    z = np.abs((features - features.mean()) / features.std())
    out = (z > 3).any(axis=1)

    result = pd.DataFrame({
        "outlier": out.astype(bool)
    }, index=features.index)

    return result
