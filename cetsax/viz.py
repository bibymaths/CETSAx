"""
viz.py
------

High-level plotting helpers for CETSA NADPH analysis:

I. Enrichment / pathway effects
J. Redox axes & roles
K. Latent factor models (PCA / FA)
L. Mixture model clusters

These are designed as thin wrappers around:

    - enrichment.summarize_pathway_effects / enrich_*(...)
    - redox.build_redox_axes / summarize_redox_by_pathway(...)
    - latent.fit_pca / fit_factor_analysis(...)
    - mixture.assign_mixture_clusters(...)

All functions accept an optional `ax` and return (fig, ax).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# I. PATHWAY EFFECTS & ENRICHMENT
# ============================================================

def plot_pathway_effects_bar(
        path_df: pd.DataFrame,
        metric: str = "NSS_mean",
        top_n: int = 20,
        ax: Optional[plt.Axes] = None,
):
    """
    Horizontal barplot of pathway-level effects.

    Parameters
    ----------
    path_df : DataFrame
        Output of summarize_pathway_effects, must contain:
            'pathway' (or custom) and the chosen metric.

    metric : str
        Column name to plot, e.g. 'NSS_mean', 'delta_max_median'.

    top_n : int
        Number of top pathways to show.

    ax : Axes or None
        If None, creates a new figure/axis.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    col_path = [c for c in path_df.columns if c.lower().startswith("path")][0]
    df = path_df.sort_values(metric, ascending=False).head(top_n)

    ax.barh(df[col_path], df[metric])
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_ylabel(col_path)
    ax.set_title(f"Top {top_n} pathways by {metric}")

    return fig, ax


def plot_pathway_enrichment_volcano(
        enr_df: pd.DataFrame,
        ax: Optional[plt.Axes] = None,
        label_top_n: int = 10,
):
    """
    Simple 'volcano-style' plot for pathway enrichment:
    x = log2(odds_ratio), y = -log10(qval).

    Works with output of enrich_overrepresentation(...).

    Parameters
    ----------
    enr_df : DataFrame
        Must contain: 'odds_ratio', 'qval', pathway column.

    label_top_n : int
        Label top_n most significant pathways.

    Returns
    -------
    fig, ax
    """
    if enr_df.empty:
        raise ValueError("enr_df is empty; nothing to plot.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    col_path = [c for c in enr_df.columns if c.lower().startswith("path")][0]

    x = np.log2(enr_df["odds_ratio"].replace(0, np.nan))
    y = -np.log10(enr_df["qval"].replace(0, np.nan))

    ax.scatter(x, y, s=20, alpha=0.7)
    ax.set_xlabel("log2(odds ratio)")
    ax.set_ylabel("-log10(q-value)")
    ax.set_title("Pathway over-representation")

    # Label top_n points by smallest qval
    top = enr_df.sort_values("qval").head(label_top_n)
    for _, row in top.iterrows():
        xx = np.log2(row["odds_ratio"]) if row["odds_ratio"] > 0 else 0.0
        yy = -np.log10(row["qval"]) if row["qval"] > 0 else 0.0
        ax.text(xx, yy, str(row[col_path]), fontsize=8)

    return fig, ax


# ============================================================
# J. REDOX AXES & ROLES
# ============================================================

def plot_redox_axes_scatter(
        redox_df: pd.DataFrame,
        x_axis: str = "axis_direct",
        y_axis: str = "axis_indirect",
        color_by: str = "redox_role",
        ax: Optional[plt.Axes] = None,
):
    """
    Scatter of redox axes (e.g. direct vs indirect), colored by role.

    Parameters
    ----------
    redox_df : DataFrame
        Output of build_redox_axes, must contain x_axis, y_axis, color_by.

    x_axis, y_axis : str
        Columns to use as x and y coordinates.

    color_by : str
        Categorical column used for coloring (e.g. 'redox_role').

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    cats = redox_df[color_by].astype(str)
    uniq = cats.unique()

    for cat in uniq:
        sub = redox_df[cats == cat]
        ax.scatter(sub[x_axis], sub[y_axis], s=15, alpha=0.6, label=str(cat))

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f"{x_axis} vs {y_axis} by {color_by}")
    ax.legend(title=color_by, fontsize=8)

    return fig, ax


def plot_redox_role_composition(
        path_redox_df: pd.DataFrame,
        top_n: int = 20,
        ax: Optional[plt.Axes] = None,
):
    """
    Stacked barplot of redox role composition per pathway.

    Expects output of summarize_redox_by_pathway(...), with columns:
        'pathway' + fractions:
            'frac_direct_core',
            'frac_indirect_responder',
            'frac_network_mediator',
            'frac_peripheral'
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    col_path = [c for c in path_redox_df.columns if c.lower().startswith("path")][0]

    cols_frac = [
        "frac_direct_core",
        "frac_indirect_responder",
        "frac_network_mediator",
        "frac_peripheral",
    ]
    df = path_redox_df.head(top_n).copy()
    bottom = np.zeros(len(df))

    x = np.arange(len(df))

    for c in cols_frac:
        vals = df[c].values
        ax.bar(x, vals, bottom=bottom, label=c)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(df[col_path], rotation=90)
    ax.set_ylabel("Fraction of proteins")
    ax.set_title(f"Redox role composition (top {top_n} pathways)")
    ax.legend(fontsize=8)

    return fig, ax


# ============================================================
# K. LATENT FACTORS (PCA / FA)
# ============================================================

def plot_pca_scores(
        scores_df: pd.DataFrame,
        meta_df: Optional[pd.DataFrame] = None,
        id_col: str = "id",
        color_by: Optional[str] = None,
        pc_x: str = "PC1",
        pc_y: str = "PC2",
        ax: Optional[plt.Axes] = None,
):
    """
    Scatter of PCA scores (PC1 vs PC2), optionally colored by a metadata column.

    Parameters
    ----------
    scores_df : DataFrame
        PCA scores indexed by id, columns like 'PC1', 'PC2', 'PC3', ...

    meta_df : DataFrame or None
        Optional metadata table with id_col and color_by.

    color_by : str or None
        Column in meta_df used to color points (categorical).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    df = scores_df.copy()
    df[id_col] = df.index

    if meta_df is not None and color_by is not None and color_by in meta_df.columns:
        merged = pd.merge(df, meta_df[[id_col, color_by]], on=id_col, how="left")
        cats = merged[color_by].astype(str)
        uniq = cats.unique()
        for cat in uniq:
            sub = merged[cats == cat]
            ax.scatter(sub[pc_x], sub[pc_y], s=15, alpha=0.6, label=str(cat))
        ax.legend(title=color_by, fontsize=8)
    else:
        ax.scatter(df[pc_x], df[pc_y], s=15, alpha=0.6)

    ax.set_xlabel(pc_x)
    ax.set_ylabel(pc_y)
    ax.set_title(f"{pc_x} vs {pc_y}")

    return fig, ax


def plot_factor_scores(
        scores_df: pd.DataFrame,
        meta_df: Optional[pd.DataFrame] = None,
        id_col: str = "id",
        color_by: Optional[str] = None,
        f_x: str = "F1",
        f_y: str = "F2",
        ax: Optional[plt.Axes] = None,
):
    """
    Same as plot_pca_scores, but for FactorAnalysis scores (F1, F2, ...).
    """
    return plot_pca_scores(
        scores_df=scores_df.rename(columns={f_x: "PC1", f_y: "PC2"}),
        meta_df=meta_df,
        id_col=id_col,
        color_by=color_by,
        pc_x="PC1",
        pc_y="PC2",
        ax=ax,
    )


# ============================================================
# L. MIXTURE CLUSTERS
# ============================================================

def plot_mixture_clusters_in_pca(
        pca_scores: pd.DataFrame,
        cluster_df: pd.DataFrame,
        id_col: str = "id",
        pc_x: str = "PC1",
        pc_y: str = "PC2",
        ax: Optional[plt.Axes] = None,
):
    """
    Plot mixture clusters in PCA space (PC1 vs PC2).

    Parameters
    ----------
    pca_scores : DataFrame
        PCA scores indexed by id, columns including pc_x, pc_y.

    cluster_df : DataFrame
        Output of assign_mixture_clusters, must contain id_col, 'cluster'.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    scores = pca_scores.copy()
    scores[id_col] = scores.index
    merged = pd.merge(scores, cluster_df[[id_col, "cluster"]], on=id_col, how="left")

    clusters = merged["cluster"].astype(int)
    uniq = np.sort(clusters.unique())

    for c in uniq:
        sub = merged[clusters == c]
        ax.scatter(sub[pc_x], sub[pc_y], s=15, alpha=0.6, label=f"cluster {c}")

    ax.set_xlabel(pc_x)
    ax.set_ylabel(pc_y)
    ax.set_title("Mixture clusters in PCA space")
    ax.legend(fontsize=8)

    return fig, ax


def plot_cluster_size_bar(
        cluster_df: pd.DataFrame,
        ax: Optional[plt.Axes] = None,
):
    """
    Barplot of mixture cluster sizes.

    Parameters
    ----------
    cluster_df : DataFrame
        Output of assign_mixture_clusters, must contain 'cluster'.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    counts = cluster_df["cluster"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of proteins")
    ax.set_title("Mixture cluster sizes")

    return fig, ax
