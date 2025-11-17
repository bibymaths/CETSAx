"""
CETSAlytics – CETSA-MS modelling toolkit.

This package currently implements ITDR-based binding curve fitting
(EC50, Hill, Emax) for proteome-wide NADPH CETSA data.
"""

from .config import DOSE_COLS, QC_MIN_UNIQUE_PEPTIDES, QC_MIN_PSMS, QC_MIN_COUNTNUM
from .dataio import load_cetsa_csv, apply_basic_qc
from .models import itdr_model
from .fit import fit_all_proteins
from .hits import call_hits, summarize_hits

__all__ = [
    "DOSE_COLS",
    "QC_MIN_UNIQUE_PEPTIDES",
    "QC_MIN_PSMS",
    "QC_MIN_COUNTNUM",
    "load_cetsa_csv",
    "apply_basic_qc",
    "itdr_model",
    "fit_all_proteins",
    "call_hits",
    "summarize_hits",
]
