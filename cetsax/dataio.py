"""
Data loading and basic filtering for CETSA NADPH ITDR dataset.
"""

from __future__ import annotations

from types import NoneType
from typing import Any

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

    Parameters
    ----------
    path : str
        Path to the CSV file.
    Returns
    -------
    pd.DataFrame
        Loaded dataset with appropriate columns.
    """
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    # Ensure dose columns are numeric
    for col in DOSE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # strip UniProt isoform suffixes: O00231-2 → O00231
    # df[ID_COL] = df[ID_COL].astype(str).str.replace(r"-\d+$", "", regex=True)
    return df


def apply_basic_qc(df: pd.DataFrame) -> type[NoneType[Any]]:
    """
    Apply simple QC criteria at the protein-replicate level.
    Filters proteins based on minimum unique peptides, PSMs, and count number.
    Parameters
    ----------
    df : pd.DataFrame
        Input CETSA dataset.
    Returns
    -------
    pd.DataFrame
        Filtered dataset passing QC criteria.
    """
    qc_df = df.query(
        f"{SUM_UNIPEPS_COL} >= @QC_MIN_UNIQUE_PEPTIDES and "
        f"{SUM_PSMS_COL} >= @QC_MIN_PSMS and "
        f"{COUNTNUM_COL} >= @QC_MIN_COUNTNUM"
    ).copy()
    return qc_df[[ID_COL, COND_COL] + DOSE_COLS + [SUM_UNIPEPS_COL, SUM_PSMS_COL, COUNTNUM_COL]]
