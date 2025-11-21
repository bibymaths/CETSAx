"""
enrichment.py
-------------

Pathway-level effect size integration and enrichment for NADPH CETSA data.
Build pathway/module-level summaries of NADPH responsiveness and perform
enrichment tests using:

  - Binary hit sets (e.g. strong / medium hits)
  - Continuous scores (e.g. NSS, delta_max, EC50)

"""

from __future__ import annotations

from typing import Dict, Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu


# ------------------------------------------------------------
# 0. BASIC HELPERS
# ------------------------------------------------------------

def _benjamini_hochberg(pvals: pd.Series) -> pd.Series:
    """
    Benjamini–Hochberg FDR correction.

    Parameters
    ----------
    pvals : Series
        Raw p-values.

    Returns
    -------
    Series
        Adjusted q-values (FDR).
    """
    p = pvals.values
    n = p.size
    order = np.argsort(p)
    ranks = np.arange(1, n + 1)
    q = np.empty_like(p, dtype=float)
    q[order] = p[order] * n / ranks
    # enforce monotonicity
    q[order] = np.minimum.accumulate(q[order][::-1])[::-1]
    q = np.clip(q, 0, 1)
    return pd.Series(q, index=pvals.index)


# ------------------------------------------------------------
# 1. PATHWAY-LEVEL EFFECT SIZE INTEGRATION
# ------------------------------------------------------------

def summarize_pathway_effects(
        metric_df: pd.DataFrame,
        annot_df: pd.DataFrame,
        id_col: str = "id",
        path_col: str = "pathway",
        metrics: Iterable[str] = ("NSS", "EC50", "delta_max", "R2"),
) -> pd.DataFrame:
    """
    Summarize NADPH responsiveness per pathway/module.

    metric_df : DataFrame
        Per-protein metric table, e.g. output from
        sensitivity.compute_sensitivity_scores or a custom
        per-protein summary with columns:
            id, EC50, delta_max, Hill, R2, NSS, ...

    annot_df : DataFrame
        Annotation table mapping proteins to pathways/modules.
        Must contain columns:
            id_col, path_col

    metrics : iterable of str
        Column names in metric_df to summarize per pathway.

    Returns
    -------
    DataFrame
        path_col | N_proteins | <metric>_mean | <metric>_median | <metric>_top25
    """
    # Merge metrics with annotations
    m = pd.merge(metric_df, annot_df[[id_col, path_col]], on=id_col, how="inner")

    agg_dict: Dict[str, Any] = {"N_proteins": (id_col, "nunique")}
    for mname in metrics:
        agg_dict[f"{mname}_mean"] = (mname, "mean")
        agg_dict[f"{mname}_median"] = (mname, "median")
        agg_dict[f"{mname}_top25"] = (mname, lambda x, _q=0.75: x.quantile(_q))

    out = (
        m.groupby(path_col)
        .agg(**agg_dict)
        .sort_values("N_proteins", ascending=False)
        .reset_index()
    )

    return out


# ------------------------------------------------------------
# 2. OVER-REPRESENTATION ENRICHMENT (BINARY HIT SETS)
# ------------------------------------------------------------

def enrich_overrepresentation(
        hits_df: pd.DataFrame,
        annot_df: pd.DataFrame,
        id_col: str = "id",
        path_col: str = "pathway",
        hit_col: str = "hit_class",
        strong_labels: Iterable[str] = ("strong",),
        min_genes: int = 3,
) -> pd.DataFrame:
    """
    Perform over-representation analysis for pathways using a binary hit set.
    Parameters
    ----------
    hits_df : DataFrame
        Per-protein hit classification table, must contain: id_col, hit_col.
    annot_df : DataFrame
        id-to-pathway mapping.
    id_col : str
        Column name for protein IDs.
    path_col : str
        Column name for pathway/module names.
    hit_col : str
        Column name for hit classification.
    strong_labels : iterable of str
        Labels in hit_col to consider as "hits".
    min_genes : int
        Minimum number of genes in a pathway to consider it for enrichment.
    Returns
    -------
    DataFrame
        path_col | n_path | n_hits | n_bg | odds_ratio | pval | qval
    """
    # Merge annotations
    df = pd.merge(hits_df[[id_col, hit_col]], annot_df[[id_col, path_col]],
                  on=id_col, how="inner")

    # Define hit set
    df["is_hit"] = df[hit_col].isin(strong_labels)

    # Background counts
    all_ids = df[id_col].unique()
    n_bg = len(all_ids)
    n_hits_total = int(df["is_hit"].sum())

    results = []

    for path, sub in df.groupby(path_col):
        genes_in_path = sub[id_col].nunique()
        if genes_in_path < min_genes:
            continue

        # Number of hits in this pathway
        hits_in_path = int(sub["is_hit"].sum())
        if hits_in_path == 0:
            continue

        # Contingency:
        #           hit   not_hit
        # path      a       b
        # other     c       d
        a = hits_in_path
        b = genes_in_path - a
        c = n_hits_total - a
        d = (n_bg - genes_in_path) - c
        if min(a, b, c, d) < 0:
            continue

        table = np.array([[a, b], [c, d]], dtype=int)
        try:
            odds, pval = fisher_exact(table, alternative="greater")
        except Exception:
            continue

        results.append(
            {
                path_col: path,
                "n_path": genes_in_path,
                "n_hits": a,
                "n_bg": n_bg,
                "odds_ratio": odds,
                "pval": pval,
            }
        )

    if not results:
        return pd.DataFrame(
            columns=[path_col, "n_path", "n_hits", "n_bg", "odds_ratio", "pval", "qval"]
        )

    res_df = pd.DataFrame(results)
    res_df["qval"] = _benjamini_hochberg(res_df["pval"])
    res_df = res_df.sort_values(["qval", "odds_ratio"], ascending=[True, False])

    return res_df


# ------------------------------------------------------------
# 3. CONTINUOUS ENRICHMENT (MANN–WHITNEY ON NSS / EFFECT SIZE)
# ------------------------------------------------------------

def enrich_continuous_mannwhitney(
        sens_df: pd.DataFrame,
        annot_df: pd.DataFrame,
        score_col: str = "NSS",
        id_col: str = "id",
        path_col: str = "pathway",
        min_genes: int = 3,
) -> pd.DataFrame:
    """
    Continuous enrichment per pathway using Mann–Whitney U tests.

    For each pathway, compares the distribution of `score_col` (e.g. NSS,
    delta_max, or -log10(EC50)) between proteins in the pathway vs all
    other proteins.

    Parameters
    ----------
    sens_df : DataFrame
        Per-protein sensitivity scores, must contain: id_col, score_col.
    annot_df : DataFrame
        id-to-pathway mapping.
    score_col : str
        Column name for continuous score to test.
    id_col : str
        Column name for protein IDs.
    path_col : str
        Column name for pathway/module names.
    min_genes : int
        Minimum number of genes in a pathway to consider it for enrichment.

    Returns
    -------
    DataFrame
        pathway | n_path | score_mean | score_median | U_stat | pval | qval
    """
    merged = pd.merge(sens_df[[id_col, score_col]], annot_df[[id_col, path_col]],
                      on=id_col, how="inner")

    all_scores = merged[[id_col, score_col]].drop_duplicates()
    background = all_scores.set_index(id_col)[score_col]

    results = []

    for path, sub in merged.groupby(path_col):
        path_ids = sub[id_col].unique()
        if len(path_ids) < min_genes:
            continue

        scores_in = background.loc[path_ids].values
        scores_out = background.drop(path_ids, errors="ignore").values

        if len(scores_out) < min_genes:
            continue

        try:
            stat, pval = mannwhitneyu(scores_in, scores_out, alternative="greater")
        except Exception:
            continue

        results.append(
            {
                path_col: path,
                "n_path": len(path_ids),
                f"{score_col}_mean": float(np.mean(scores_in)),
                f"{score_col}_median": float(np.median(scores_in)),
                "U_stat": float(stat),
                "pval": float(pval),
            }
        )

    if not results:
        return pd.DataFrame(
            columns=[
                path_col,
                "n_path",
                f"{score_col}_mean",
                f"{score_col}_median",
                "U_stat",
                "pval",
                "qval",
            ]
        )

    res_df = pd.DataFrame(results)
    res_df["qval"] = _benjamini_hochberg(res_df["pval"])
    res_df = res_df.sort_values(["qval", f"{score_col}_mean"], ascending=[True, False])

    return res_df
