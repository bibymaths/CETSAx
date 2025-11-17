"""
Data loading and basic filtering for CETSA NADPH ITDR dataset.
"""

from __future__ import annotations

import pandas as pd

from .config import (
    DOSE_COLS,
    ID_COL,
    COND_COL,
    SUM_UNIPEPS_COL,
    SUM_PSMS_COL,
    COUNTNUM_COL,
    QC_MIN_UNIQUE_PEPTIDES,
    QC_MIN_PSMS,
    QC_MIN_COUNTNUM,
)


def load_cetsa_csv(path: str) -> pd.DataFrame:
    """
    Load CETSA NADPH ITDR dataset from CSV.

    Assumes a column 'Unnamed: 0' can be dropped if present.
    """
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    # Ensure dose columns are numeric
    for col in DOSE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def apply_basic_qc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply simple QC criteria at the protein-replicate level.
    """
    qc_df = df.query(
        f"{SUM_UNIPEPS_COL} >= @QC_MIN_UNIQUE_PEPTIDES and "
        f"{SUM_PSMS_COL} >= @QC_MIN_PSMS and "
        f"{COUNTNUM_COL} >= @QC_MIN_COUNTNUM"
    ).copy()
    return qc_df[[ID_COL, COND_COL] + DOSE_COLS + [SUM_UNIPEPS_COL, SUM_PSMS_COL, COUNTNUM_COL]]
