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


def plot_goodness_of_fit(
        df: pd.DataFrame,
        fit_df: pd.DataFrame,
        ax: Optional[plt.Axes] = None,
):
    """
    Global goodness-of-fit plot: observed vs predicted values for all proteins,
    colored by condition (r1, r2).

    Returns:
        fig, ax : matplotlib Figure and Axes objects
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # Join raw data with fit parameters
    merged = pd.merge(
        df,
        fit_df[[ID_COL, COND_COL, "E0", "Emax", "log10_EC50", "Hill"]],
        on=[ID_COL, COND_COL],
        how="inner",
        validate="m:m",
    )

    if merged.empty:
        raise ValueError("No overlap between data and fitted parameters.")

    doses = np.array(DOSE_COLS, dtype=float)

    obs_vals = []
    pred_vals = []
    cond_labels = []

    for _, row in merged.iterrows():
        y_obs = row[DOSE_COLS].values.astype(float)

        E0 = row["E0"]
        Emax = row["Emax"]
        logEC50 = row["log10_EC50"]
        h = row["Hill"]

        y_pred = itdr_model(doses, E0, Emax, logEC50, h)

        obs_vals.append(y_obs)
        pred_vals.append(y_pred)
        cond_labels.extend([row[COND_COL]] * len(doses))

    obs_vals = np.concatenate(obs_vals)
    pred_vals = np.concatenate(pred_vals)
    cond_labels = np.array(cond_labels)

    # Condition colors
    unique_conds = pd.unique(cond_labels)
    colors = {cond: f"C{i}" for i, cond in enumerate(unique_conds)}

    for cond in unique_conds:
        m = cond_labels == cond
        ax.scatter(
            pred_vals[m], obs_vals[m],
            alpha=0.5, s=10, label=str(cond)
        )

    # 1:1 diagonal
    lo = min(obs_vals.min(), pred_vals.min())
    hi = max(obs_vals.max(), pred_vals.max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1)

    ax.set_xlabel("Predicted (model)")
    ax.set_ylabel("Observed (data)")
    ax.set_title("Goodness of Fit: All Proteins")
    ax.legend(title="Condition")

    return fig, ax
