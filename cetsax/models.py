"""
Mathematical models for ITDR CETSA curves.
"""

from __future__ import annotations

import numpy as np


def itdr_model(c: np.ndarray, E0: float, Emax: float, logEC50: float, h: float) -> np.ndarray:
    """
    4-parameter logistic ITDR model for CETSA:

        f(c) = E0 + (Emax - E0) / (1 + (EC50 / c)^h)

    where EC50 = 10 ** logEC50 (parameterized in log10 space for stability).
    """
    EC50 = 10.0 ** logEC50
    # Avoid division by zero
    c = np.asarray(c, dtype=float)
    c_safe = np.where(c <= 0, np.min(c[c > 0]) * 1e-3, c)
    return E0 + (Emax - E0) / (1.0 + (EC50 / c_safe) ** h)
