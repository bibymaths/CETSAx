"""
viz_hits.py
-----------

Hit-calling and visualization functions for CETSA ITDR data.
This module provides functions to classify hits based on EC50, delta_max,
and R2 thresholds, generate ranked hit tables, and create diagnostic plots
to visualize hit characteristics and replicate consistency.
"""
# BSD 3-Clause License
#
# Copyright (c) 2025, Abhinav Mishra
# All rights reserved.
# Email: mishraabhinav36@gmail.com
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of Abhinav Mishra nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Hit classification logic
# ---------------------------------------------------------------------

def classify_hit(
        row: pd.Series,
        ec50_strong: float = 0.01,
        ec50_medium: float = 0.5,
        delta_strong: float = 0.10,
        delta_medium: float = 0.08,
        r2_strong: float = 0.70,
        r2_medium: float = 0.50,
) -> str:
    """
    Classify a single (EC50, delta_max, R2) triplet into strong/medium/weak.

    Thresholds are configurable but default to:
        strong: EC50 < 0.01, delta_max > 0.10, R2 > 0.70
        medium: EC50 < 0.5,  delta_max > 0.08, R2 > 0.50
        else:  weak
    """
    ec50 = float(row["EC50"])
    dm = float(row["delta_max"])
    r2 = float(row["R2"])

    if (ec50 < ec50_strong) and (dm > delta_strong) and (r2 > r2_strong):
        return "strong"
    elif (ec50 < ec50_medium) and (dm > delta_medium) and (r2 > r2_medium):
        return "medium"
    else:
        return "weak"


def build_hits_table(
        fits_df: pd.DataFrame,
        ec50_strong: float = 0.01,
        ec50_medium: float = 0.5,
        delta_strong: float = 0.10,
        delta_medium: float = 0.08,
        r2_strong: float = 0.70,
        r2_medium: float = 0.50,
        id_col: str = "id",
        cond_col: str = "condition",
) -> pd.DataFrame:
    """
    Add hit_class per row and aggregate to a per-protein ranked table.

    Returns a DataFrame with columns:
        id, n_reps, class_counts, EC50_median, delta_max_median, R2_median, dominant_class
    """
    df = fits_df.copy()

    df["hit_class"] = df.apply(
        classify_hit,
        axis=1,
        ec50_strong=ec50_strong,
        ec50_medium=ec50_medium,
        delta_strong=delta_strong,
        delta_medium=delta_medium,
        r2_strong=r2_strong,
        r2_medium=r2_medium,
    )

    agg = (
        df.groupby(id_col)
        .agg(
            n_reps=(cond_col, "nunique"),
            class_counts=("hit_class", lambda x: x.value_counts().to_dict()),
            EC50_median=("EC50", "median"),
            delta_max_median=("delta_max", "median"),
            R2_median=("R2", "median"),
        )
        .reset_index()
    )

    def dominant_class(d):
        if not isinstance(d, dict):
            return "weak"
        strong = d.get("strong", 0)
        medium = d.get("medium", 0)
        weak = d.get("weak", 0)
        if strong > 0:
            return "strong"
        if medium > 0:
            return "medium"
        return "weak"

    agg["dominant_class"] = agg["class_counts"].apply(dominant_class)

    class_rank = {"strong": 0, "medium": 1, "weak": 2}
    agg["class_rank"] = agg["dominant_class"].map(class_rank)

    agg_sorted = agg.sort_values(
        ["class_rank", "delta_max_median", "R2_median"],
        ascending=[True, False, False],
    ).drop(columns=["class_rank"])

    return agg_sorted


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def plot_ec50_vs_delta(
        df: pd.DataFrame,
        ec50_cut: float = 0.01,
        delta_cut: float = 0.10,
        ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    EC50 vs delta_max scatter with quadrant lines.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.scatter(df["EC50"], df["delta_max"], s=10, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("EC50 (M, log scale)")
    ax.set_ylabel("delta_max")
    ax.set_title("EC50 vs delta_max")

    ax.axvline(ec50_cut)
    ax.axhline(delta_cut)

    return fig, ax


def plot_ec50_replicates(
        df: pd.DataFrame,
        id_col: str = "id",
        cond_col: str = "condition",
        cond_r1: str = "NADPH.r1",
        cond_r2: str = "NADPH.r2",
        ax: Optional[plt.Axes] = None,
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    EC50 replicate consistency plot: EC50_r1 vs EC50_r2 (log-log).

    Returns (fig, ax) or None if required conditions are missing.
    """
    pivot = df.pivot_table(
        index=id_col, columns=cond_col, values="EC50", aggfunc="first"
    )
    if {cond_r1, cond_r2}.issubset(pivot.columns):
        sub = pivot[[cond_r1, cond_r2]].dropna()
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.scatter(sub[cond_r1], sub[cond_r2], s=10, alpha=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"EC50 {cond_r1} (M)")
        ax.set_ylabel(f"EC50 {cond_r2} (M)")
        ax.set_title("Replicate consistency: EC50")

        lo = float(sub.min().min())
        hi = float(sub.max().max())
        ax.plot([lo, hi], [lo, hi])

        return fig, ax

    return None


def plot_r2_vs_delta(
        df: pd.DataFrame,
        ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    R2 vs delta_max scatter.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.scatter(df["delta_max"], df["R2"], s=10, alpha=0.5)
    ax.set_xlabel("delta_max")
    ax.set_ylabel("R2")
    ax.set_title("R2 vs delta_max")

    return fig, ax


def plot_ec50_vs_r2(
        df: pd.DataFrame,
        ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    EC50 vs R2 scatter (log-scale EC50).
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.scatter(df["EC50"], df["R2"], s=10, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("EC50 (M, log scale)")
    ax.set_ylabel("R2")
    ax.set_title("EC50 vs R2")

    return fig, ax


# ---------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------

def run_hit_calling_and_plots(
        fits_df: pd.DataFrame,
        out_dir: Path | str,
        id_col: str = "id",
        cond_col: str = "condition",
        ec50_strong: float = 0.01,
        ec50_medium: float = 0.5,
        delta_strong: float = 0.10,
        delta_medium: float = 0.08,
        r2_strong: float = 0.70,
        r2_medium: float = 0.50,
) -> Dict[str, str | pd.DataFrame]:
    """
    Full hit-calling + plotting pipeline.

    Parameters
    ----------
    fits_df : DataFrame
        Output of fit_all_proteins with columns:
        id, condition, EC50, delta_max, R2, ...

    out_dir : Path or str
        Directory where plots and ranked hits table will be saved.

    Returns
    -------
    dict with keys:
        - 'hits_table': ranked hits DataFrame
        - 'ec50_vs_delta': path to EC50 vs delta_max plot
        - 'ec50_r1_vs_r2': path (if generated)
        - 'r2_vs_delta': path
        - 'ec50_vs_r2': path
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build hits table
    hits_df = build_hits_table(
        fits_df,
        ec50_strong=ec50_strong,
        ec50_medium=ec50_medium,
        delta_strong=delta_strong,
        delta_medium=delta_medium,
        r2_strong=r2_strong,
        r2_medium=r2_medium,
        id_col=id_col,
        cond_col=cond_col,
    )
    hits_path = out_dir / "cetsa_hits_ranked.csv"
    hits_df.to_csv(hits_path, index=False)

    paths: Dict[str, str | pd.DataFrame] = {}
    paths["hits_table"] = hits_df

    # 2) EC50 vs delta_max
    fig, ax = plot_ec50_vs_delta(
        fits_df,
        ec50_cut=ec50_strong,
        delta_cut=delta_strong,
    )
    p_ec50_delta = out_dir / "ec50_vs_delta_max.png"
    fig.savefig(p_ec50_delta, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths["ec50_vs_delta"] = str(p_ec50_delta)

    # 3) EC50_r1 vs EC50_r2 (if applicable)
    rep = plot_ec50_replicates(
        fits_df,
        id_col=id_col,
        cond_col=cond_col,
        cond_r1="NADPH.r1",
        cond_r2="NADPH.r2",
    )
    if rep is not None:
        fig, ax = rep
        p_rep = out_dir / "ec50_r1_vs_r2.png"
        fig.savefig(p_rep, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths["ec50_r1_vs_r2"] = str(p_rep)

    # 4) R2 vs delta_max
    fig, ax = plot_r2_vs_delta(fits_df)
    p_r2_delta = out_dir / "r2_vs_delta_max.png"
    fig.savefig(p_r2_delta, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths["r2_vs_delta"] = str(p_r2_delta)

    # 5) EC50 vs R2
    fig, ax = plot_ec50_vs_r2(fits_df)
    p_ec50_r2 = out_dir / "ec50_vs_r2.png"
    fig.savefig(p_ec50_r2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths["ec50_vs_r2"] = str(p_ec50_r2)

    return paths
