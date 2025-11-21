"""
hits.py
------------
Hit calling and summary for CETSA ITDR binding models.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from .config import ID_COL, COND_COL, DOSE_COLS


def call_hits(
        fit_df: pd.DataFrame,
        r2_min: float = 0.8,
        delta_min: float = 0.1,
) -> pd.DataFrame:
    """
    Filter fitted curves to keep only high-confidence hits.

    Parameters
    ----------
    fit_df : pd.DataFrame
        Data Frame containing fitted curve parameters with at least the following columns:
        - 'R2': Coefficient of determination of the fit.
        - 'delta_max': Maximum change in response.
        - 'EC50': Half-maximal effective concentration.
    r2_min : float
        Minimum R² value to consider a fit as a hit.
    delta_min : float
        Minimum delta_max value to consider a fit as a hit.
    Returns
    -------
    pd.DataFrame
        Data Frame containing only the hits that meet the specified criteria.
    """
    if fit_df.empty:
        return fit_df.copy()

    doses = np.array(DOSE_COLS, dtype=float)
    ec50_min, ec50_max = doses.min(), doses.max()

    hits = (
        fit_df
        .query("R2 >= @r2_min and delta_max >= @delta_min")
        .query("EC50 >= @ec50_min and EC50 <= @ec50_max")
        .copy()
    )
    return hits


def summarize_hits(hits_df: pd.DataFrame, min_reps: int = 2) -> pd.DataFrame:
    """
    Aggregate hit information at the protein level across replicates.
    Parameters
    ----------
    hits_df : pd.DataFrame
        Data Frame containing hit information with at least the following columns:
        - ID_COL: Identifier for the protein or target.
        - COND_COL: Condition or replicate identifier.
        - 'EC50': Half-maximal effective concentration.
        - 'Emax': Maximum effect.
        - 'Hill': Hill coefficient.
    min_reps : int
        Minimum number of replicates required to include a protein in the summary.
    Returns
    -------
    pd.DataFrame
        Summary Data Frame with aggregated hit information per protein.
    """
    if hits_df.empty:
        return pd.DataFrame(
            columns=[
                ID_COL,
                "n_reps",
                "EC50_median",
                "EC50_sd",
                "Emax_median",
                "Hill_median",
            ]
        )

    summary = (
        hits_df
        .groupby(ID_COL, dropna=False)
        .agg(
            n_reps=(COND_COL, "nunique"),
            EC50_median=("EC50", "median"),
            EC50_sd=("EC50", "std"),
            Emax_median=("Emax", "median"),
            Hill_median=("Hill", "median"),
        )
        .reset_index()
    )

    summary = summary.query("n_reps >= @min_reps").copy()
    return summary
