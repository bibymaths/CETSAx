"""
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

    - r2_min: minimum R^2 for a good fit
    - delta_min: minimum max change in signal across doses
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

    - min_reps: minimum distinct conditions (replicates) where the protein is a hit.
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
