"""
Quick plotting utilities for CETSA ITDR curves.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import DOSE_COLS, ID_COL, COND_COL
from .models import itdr_model


def plot_protein_curve(
    df: pd.DataFrame,
    fit_df: pd.DataFrame,
    protein_id: str,
    condition: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot raw ITDR data and fitted curve for a given protein (and condition).
    """
    if ax is None:
        _, ax = plt.subplots()

    doses = np.array(DOSE_COLS, dtype=float)

    subset = df[df[ID_COL] == protein_id]
    if condition is not None:
        subset = subset[subset[COND_COL] == condition]

    if subset.empty:
        raise ValueError(f"No data for id={protein_id!r}, condition={condition!r}")

    row = subset.iloc[0]
    y = row[DOSE_COLS].values.astype(float)

    # Plot observed points
    ax.scatter(doses, y, label="Observed", marker="o")

    # Plot fit if available
    fsub = fit_df[(fit_df[ID_COL] == protein_id)]
    if condition is not None:
        fsub = fsub[fsub[COND_COL] == condition]

    if not fsub.empty:
        pars = fsub.iloc[0]
        E0, Emax, logEC50, h = pars["E0"], pars["Emax"], pars["log10_EC50"], pars["Hill"]
        x_grid = np.logspace(np.log10(doses.min()), np.log10(doses.max()), 200)
        y_grid = itdr_model(x_grid, E0, Emax, logEC50, h)
        ax.plot(x_grid, y_grid, label="Fit", linestyle="-")
        ax.set_xscale("log")

    ax.set_xlabel("NADPH concentration (M)")
    ax.set_ylabel("Scaled soluble fraction")
    ax.set_title(f"{protein_id} ({condition})")
    ax.legend()
    return ax

