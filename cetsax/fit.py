"""
Fitting EC50 / KD surrogate ITDR models per protein and per replicate.
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .config import DOSE_COLS, ID_COL, COND_COL
from .models import itdr_model


def _fit_single_curve(doses: np.ndarray, y: np.ndarray) -> Dict[str, Any] | None:
    """
    Fit ITDR logistic model to a single dose-response vector.

    Returns a dict of parameters and diagnostics or None if fit fails or no effect.
    """
    # Skip if basically flat
    if np.nanmax(np.abs(y - 1.0)) < 0.05:
        return None

    # Initial guesses
    E0_init = 1.0
    Emax_init = float(np.nanmax(y))
    logEC50_init = math.log10(np.median(doses))
    h_init = 1.0
    p0 = [E0_init, Emax_init, logEC50_init, h_init]

    # Bounds
    lower_bounds = [0.5, 0.5, math.log10(doses.min()) - 2.0, 0.1]
    upper_bounds = [1.5, 5.0, math.log10(doses.max()) + 2.0, 5.0]

    try:
        popt, pcov = curve_fit(
            itdr_model,
            doses,
            y,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000,
        )
    except Exception:
        return None

    E0, Emax, logEC50, h = popt
    EC50 = 10.0 ** logEC50

    y_pred = itdr_model(doses, *popt)
    rss = float(np.sum((y - y_pred) ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else np.nan

    delta_max = float(np.nanmax(y) - np.nanmin(y))

    return {
        "E0": float(E0),
        "Emax": float(Emax),
        "EC50": float(EC50),
        "log10_EC50": float(logEC50),
        "Hill": float(h),
        "R2": r2,
        "delta_max": delta_max,
    }


def fit_all_proteins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit ITDR curves for all proteins and replicates in a QC checked dataframe.
    """
    doses = np.array(DOSE_COLS, dtype=float)

    # Materialize groups so we can iterate in parallel and know length for tqdm
    grouped_items: List[Tuple[Tuple[Any, Any], pd.DataFrame]] = list(
        df.groupby([ID_COL, COND_COL], dropna=False)
    )

    def _process_group(item: Tuple[Tuple[Any, Any], pd.DataFrame]) -> Dict[str, Any] | None:
        (pid, cond), grp = item
        row = grp.iloc[0]
        y = row[DOSE_COLS].values.astype(float)

        fit_res = _fit_single_curve(doses, y)
        if fit_res is None:
            return None

        fit_res[ID_COL] = pid
        fit_res[COND_COL] = cond
        return fit_res

    # Parallel execution with progress bar
    results = Parallel(n_jobs=-1)(
        delayed(_process_group)(item)
        for item in tqdm(grouped_items, desc="Fitting proteins", unit="prot")
    )

    # Drop failed / None fits
    results = [r for r in results if r is not None]

    if not results:
        return pd.DataFrame(
            columns=[ID_COL, COND_COL, "E0", "Emax", "EC50", "log10_EC50", "Hill", "R2", "delta_max"]
        )

    res_df = pd.DataFrame(results)
    return res_df[
        [ID_COL, COND_COL, "E0", "Emax", "EC50", "log10_EC50", "Hill", "R2", "delta_max"]
    ]