"""
fit.py
-------------
ITDR curve fitting for CETSA data.
Fit ITDR curves to dose-response data using a robust logistic model
with monotonic smoothing and Hill slope regularization.

"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Hashable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from sklearn.isotonic import IsotonicRegression

from .config import DOSE_COLS, ID_COL, COND_COL


def _itdr_model_log(
        logc: np.ndarray,
        E0: float,
        Emax: float,
        logEC50: float,
        h: float,
) -> np.ndarray:
    """
    ITDR model parameterized in log10(concentration) space:

        f(logc) = E0 + (Emax - E0) / (1 + 10^((logEC50 - logc) * h))

    This is numerically more stable than working in linear [c] space
    when doses span many orders of magnitude.
    Parameters:
        logc: log10(concentration) values
        E0: baseline effect
        Emax: max effect
        logEC50: log10(EC50) value
        h: Hill slope
    Returns:
        Predicted effect values at given logc
    """
    logc = np.asarray(logc, dtype=float)
    exponent = (logEC50 - logc) * h
    denom = 1.0 + np.power(10.0, exponent)
    return E0 + (Emax - E0) / denom


def _fit_single_curve(doses: np.ndarray, y: np.ndarray) -> Dict[str, Any] | None:
    """
    Fit ITDR logistic model (in log-dose space) to a single dose-response vector.

    Parameters:
        doses: 1D array of dose values (same length as y)
        y: 1D array of observed effect values
    Returns:
        Dict of fit parameters and diagnostics, or None if fit failed / invalid.
    """
    y = np.asarray(y, dtype=float)
    doses = np.asarray(doses, dtype=float)

    # Mask invalid points (NaN / inf)
    mask = np.isfinite(y) & np.isfinite(doses)
    if mask.sum() < 4:
        return None

    doses_valid = doses[mask]
    y_valid = y[mask]

    # Basic variance check: skip practically flat curves
    if np.nanstd(y_valid) < 0.015:
        return None

    # Work in log10-dose space
    logd_valid = np.log10(doses_valid)

    # ------------------------------------------------------------------
    # 1) Monotonic smoothing via isotonic regression
    # ------------------------------------------------------------------
    # Decide direction: stabilization (up) or destabilization (down)
    direction = 1.0 if y_valid[-1] >= y_valid[0] else -1.0
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    if direction > 0:
        y_smooth = iso.fit_transform(logd_valid, y_valid)
    else:
        # Fit isotonic on -y and flip back
        y_smooth = -iso.fit_transform(logd_valid, -y_valid)

    # After smoothing, re-check variance
    if np.nanstd(y_smooth) < 0.015:
        return None

    # ------------------------------------------------------------------
    # 2) Initial guesses
    # ------------------------------------------------------------------
    # Baseline from first two points (smoothed)
    E0_init = float(np.mean(y_smooth[:2]))

    # Max effect from last two points (smoothed)
    Emax_init = float(np.mean(y_smooth[-2:]))

    # Midpoint-based EC50 guess: where y is halfway between min and max
    y_min = float(np.nanmin(y_smooth))
    y_max = float(np.nanmax(y_smooth))
    y_mid = 0.5 * (y_min + y_max)
    idx_mid = int(np.argmin(np.abs(y_smooth - y_mid)))
    logEC50_init = float(logd_valid[idx_mid])

    # Hill slope: mild cooperativity as a starting point
    h_init = 1.0

    p0 = np.array([E0_init, Emax_init, logEC50_init, h_init], dtype=float)

    # ------------------------------------------------------------------
    # 3) Bounds
    # ------------------------------------------------------------------
    # E0 and Emax: allow small deviations; cap Emax at 3 to kill crazy fits
    E0_low, E0_high = 0.5, 1.5
    Emax_low, Emax_high = 0.5, 3.0

    # logEC50 bounds: narrow around observed dose range
    logd_min = float(logd_valid.min())
    logd_max = float(logd_valid.max())
    logEC50_low = logd_min - 0.3
    logEC50_high = logd_max + 0.3

    # Hill: allow shallow to moderately steep, but not insane
    h_low, h_high = 0.1, 6.0

    lower_bounds = np.array([E0_low, Emax_low, logEC50_low, h_low], dtype=float)
    upper_bounds = np.array([E0_high, Emax_high, logEC50_high, h_high], dtype=float)

    # ------------------------------------------------------------------
    # 4) Weights + Hill penalty in residuals
    # ------------------------------------------------------------------
    # Dose-based weights: emphasize low concentrations.
    alpha = 0.3
    sigma = np.power(doses_valid, alpha)
    sigma[sigma <= 0] = np.min(sigma[sigma > 0]) * 0.1

    # Hill penalty strength and target
    lambda_hill = 0.05  # tune: larger → stronger push toward h_target
    h_target = 1.0

    def residuals(params: np.ndarray) -> np.ndarray:
        E0, Emax, logEC50, h = params
        y_pred = _itdr_model_log(logd_valid, E0, Emax, logEC50, h)

        # Data residuals (weighted)
        res_data = (y_smooth - y_pred) / sigma

        # Single penalty term for Hill (same scale as residuals)
        res_pen = np.array([lambda_hill * (h - h_target)], dtype=float)

        # Concatenate: least_squares will apply soft-L1 over all components
        return np.concatenate([res_data, res_pen])

    try:
        res = least_squares(
            residuals,
            p0,
            bounds=(lower_bounds, upper_bounds),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=20000,
            method="trf",
        )
    except Exception:
        return None

    if not res.success:
        return None

    E0, Emax, logEC50, h = map(float, res.x)
    EC50 = float(10.0 ** logEC50)

    # ------------------------------------------------------------------
    # 5) Diagnostics on original (unsmoothed) data
    # ------------------------------------------------------------------
    y_pred_valid = _itdr_model_log(logd_valid, E0, Emax, logEC50, h)

    rss = float(np.sum((y_valid - y_pred_valid) ** 2))
    tss = float(np.sum((y_valid - np.mean(y_valid)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else np.nan

    delta_max = float(np.nanmax(y_valid) - np.nanmin(y_valid))

    # Drop obviously useless fits
    if r2 < 0.1 or delta_max < 0.05:
        return None

    return {
        "E0": E0,
        "Emax": Emax,
        "EC50": EC50,
        "log10_EC50": logEC50,
        "Hill": h,
        "R2": r2,
        "delta_max": delta_max,
    }


def fit_all_proteins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit ITDR curves for all proteins and replicates in a QC-checked dataframe.
    Parameters:
        df: DataFrame
            Input data with columns:
                ID_COL, COND_COL, DOSE_COLS...
    Returns:
        DataFrame
            Fit results with columns:
                ID_COL, COND_COL, E0, Emax, EC50, log10_EC50, Hill, R2, delta_max
    """
    doses = np.array(DOSE_COLS, dtype=float)

    # Materialize groups so we can iterate in parallel and know length for tqdm
    grouped_items: List[Tuple[Hashable, pd.DataFrame]] = list(
        df.groupby([ID_COL, COND_COL], dropna=False)
    )

    def _process_group(item: Tuple[Hashable, pd.DataFrame]) -> Dict[str, Any] | None:
        """
        Process a single (protein, condition) group and fit the curve.
        Parameters:
            item: Tuple of (group key, group DataFrame)
        Returns:
            Dict of fit results or None if fit failed.
        """
        key, grp = item
        pid, cond = key
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
            columns=[
                ID_COL,
                COND_COL,
                "E0",
                "Emax",
                "EC50",
                "log10_EC50",
                "Hill",
                "R2",
                "delta_max",
            ]
        )

    res_df = pd.DataFrame(results)
    return res_df[
        [ID_COL, COND_COL, "E0", "Emax", "EC50", "log10_EC50", "Hill", "R2", "delta_max"]
    ]
