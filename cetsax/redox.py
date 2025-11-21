"""
redox.py
--------

Redox-axis reconstruction from NADPH CETSA data.
Defines functions to build redox axes per protein based on
sensitivity scores, hit classifications, and network centrality,
as well as to summarize redox roles at the pathway level.

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

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


# ------------------------------------------------------------
# 0. HELPER: robust scaling to [0, 1]
# ------------------------------------------------------------

def _robust_scale_01(series: pd.Series) -> pd.Series:
    """
    Robustly scale a 1D series to [0, 1] using median/IQR + logistic squash.
    High raw value -> high score (0..1).

    Parameters
    ----------
    series : pd.Series
        Input data series.
    Returns
    -------
    pd.Series
        Scaled series in [0, 1].
    """
    vals = series.values.reshape(-1, 1)
    if len(series) == 0:
        return series.copy()

    rs = RobustScaler()
    try:
        z = rs.fit_transform(vals).flatten()
    except Exception:
        z = np.zeros_like(series.values)

    z = 1 / (1 + np.exp(-z))  # logistic to (0,1)
    return pd.Series(z, index=series.index)


def _inv_scale_01(series: pd.Series) -> pd.Series:
    """
    Invert robust scale: low raw value -> high score (0..1).
    Parameters
    ----------
    series : pd.Series
        Input data series.
    Returns
    -------
    pd.Series
        Inverted scaled series in [0, 1].
    """
    s = _robust_scale_01(series)
    return 1.0 - s


# ------------------------------------------------------------
# 1. CORE: build redox axes per protein
# ------------------------------------------------------------

def build_redox_axes(
        fits_df: pd.DataFrame,
        sens_df: pd.DataFrame,
        hits_df: pd.DataFrame,
        net_df: Optional[pd.DataFrame] = None,
        id_col: str = "id",
        hit_col: str = "dominant_class",
        degree_col: str = "degree",
        betweenness_col: str = "betweenness",
) -> pd.DataFrame:
    """
    Build redox axes for each protein.

    Inputs
    ------
    fits_df : DataFrame
        Per-protein or per-(id,condition) CETSA parameters; must contain:
            id_col, EC50, delta_max, R2, Hill
        If per-condition, should be pre-aggregated to protein-level
        (e.g. median across replicates) before calling this function.

    sens_df : DataFrame
        Output of sensitivity.compute_sensitivity_scores with columns:
            id_col, NSS, EC50, delta_max, Hill, R2, ...

    hits_df : DataFrame
        Per-protein hit classification; must contain:
            id_col, hit_col (e.g. "strong", "medium", "weak")

    net_df : DataFrame, optional
        Network centrality stats; if provided, must contain:
            id_col, degree_col, betweenness_col
        If None, network centrality will be ignored.
    id_col : str
        Column name for protein identifier.
    hit_col : str
        Column name for hit classification.
    degree_col : str
        Column name for network degree centrality.
    betweenness_col : str
        Column name for network betweenness centrality.

    Returns
    -------
    DataFrame
        id | EC50 | delta_max | R2 | NSS |
        axis_direct | axis_indirect | axis_network |
        redox_role
    """
    # Merge core tables on id
    base = pd.merge(
        sens_df[[id_col, "EC50", "delta_max", "R2", "NSS"]],
        hits_df[[id_col, hit_col]],
        on=id_col,
        how="left",
    )

    # Attach network centrality if provided
    if net_df is not None:
        net_cols = [c for c in [id_col, degree_col, betweenness_col] if c in net_df.columns]
        base = pd.merge(base, net_df[net_cols], on=id_col, how="left")
    else:
        base[degree_col] = np.nan
        base[betweenness_col] = np.nan

    base = base.drop_duplicates(subset=[id_col]).set_index(id_col)

    # --------------------------------------------------------
    # 1A. Build scaled components
    # --------------------------------------------------------
    # 1 / EC50: higher score -> more sensitive
    ec50_inv = _inv_scale_01(base["EC50"])

    # Effect size
    dm_scaled = _robust_scale_01(base["delta_max"])

    # Curve quality
    r2_scaled = _robust_scale_01(base["R2"])

    # Overall sensitivity (NSS) is already composite; rescale for safety
    nss_scaled = _robust_scale_01(base["NSS"])

    # Network centrality combined; if NaN, becomes low
    deg_scaled = _robust_scale_01(base[degree_col].fillna(0.0))
    bet_scaled = _robust_scale_01(base[betweenness_col].fillna(0.0))
    centrality = 0.5 * deg_scaled + 0.5 * bet_scaled

    # Hit class weights
    hit = base[hit_col].fillna("weak").astype(str)
    w_strong = (hit == "strong").astype(float)
    w_medium = (hit == "medium").astype(float)
    w_weak = (hit == "weak").astype(float)

    # --------------------------------------------------------
    # 1B. Axes definitions
    # --------------------------------------------------------
    # Axis 1: direct NADPH core
    #   - High when: EC50 low, delta_max high, R2 good, NSS high, hit strong.
    axis_direct = (
            0.35 * ec50_inv +
            0.25 * dm_scaled +
            0.15 * r2_scaled +
            0.25 * nss_scaled
    )
    # Boost for "strong" hits, dampen for weak
    axis_direct = axis_direct * (1.0 + 0.5 * w_strong - 0.2 * w_weak)

    # Axis 2: indirect / downstream responder
    #   - High when: NSS moderate, EC50 not extremely low,
    #                delta_max moderate, medium hit class.
    #   EC50_midscore emphasizes mid-range EC50 (not too low, not too high).
    ec50_mid = ec50_inv.copy()
    ec50_mid = np.minimum(ec50_mid, 1.0 - ec50_mid)  # highest in midrange

    axis_indirect = (
            0.30 * ec50_mid +
            0.30 * dm_scaled +
            0.20 * nss_scaled +
            0.20 * r2_scaled
    )
    axis_indirect = axis_indirect * (1.0 + 0.5 * w_medium)

    # Axis 3: network mediator
    #   - High when: centrality high, NSS moderate, not extremely strong direct.
    axis_network = (
            0.6 * centrality +
            0.25 * nss_scaled +
            0.15 * r2_scaled
    )
    # Downweight pure strong-core binders; upweight mediums
    axis_network = axis_network * (1.0 + 0.3 * w_medium - 0.3 * w_strong)

    # Normalize axes to [0,1] for interpretability
    def _norm01(s: pd.Series) -> pd.Series:
        if s.empty:
            return s
        v = s.values.astype(float)
        v_min = np.nanmin(v)
        v_max = np.nanmax(v)
        if v_max <= v_min:
            return pd.Series(np.zeros_like(v), index=s.index)
        return pd.Series((v - v_min) / (v_max - v_min), index=s.index)

    axis_direct_n = _norm01(axis_direct)
    axis_indirect_n = _norm01(axis_indirect)
    axis_network_n = _norm01(axis_network)

    base["axis_direct"] = axis_direct_n
    base["axis_indirect"] = axis_indirect_n
    base["axis_network"] = axis_network_n

    # --------------------------------------------------------
    # 1C. Assign redox roles per protein
    # --------------------------------------------------------
    def _assign_role(row):
        """
        Assign redox role based on axis scores.
        Parameters
        ----------
        row : pd.Series
            Row of base DataFrame with axis scores.
        Returns
        -------
        str
            Assigned redox role.
        """
        d = row["axis_direct"]
        ind = row["axis_indirect"]
        net = row["axis_network"]

        # simple rules; can be refined
        if d >= 0.7 and d >= ind and d >= net:
            return "direct_core"
        if net >= 0.7 and net >= d and net >= ind:
            return "network_mediator"
        if ind >= 0.6 and ind >= d and ind >= net:
            return "indirect_responder"
        if max(d, ind, net) < 0.3:
            return "peripheral"
        return "mixed"

    base["redox_role"] = base.apply(_assign_role, axis=1)

    return base.reset_index()


# ------------------------------------------------------------
# 2. PATHWAY-LEVEL REDOX SUMMARIES
# ------------------------------------------------------------

def summarize_redox_by_pathway(
        redox_df: pd.DataFrame,
        annot_df: pd.DataFrame,
        id_col: str = "id",
        path_col: str = "pathway",
) -> pd.DataFrame:
    """
    Summarize redox axes and roles per pathway/module.

    redox_df : DataFrame
        Output of build_redox_axes, with columns:
            id_col, axis_direct, axis_indirect, axis_network, redox_role, ...

    annot_df : DataFrame
        id-to-pathway mapping.
        Must contain: id_col, path_col
    id_col : str
        Column name for protein identifier.
    path_col : str
        Column name for pathway/module annotation.

    Returns
    -------
    DataFrame
        pathway | N | direct_mean | indirect_mean | network_mean |
        frac_direct | frac_indirect | frac_network | frac_peripheral
    """
    merged = pd.merge(
        redox_df,
        annot_df[[id_col, path_col]],
        on=id_col,
        how="inner",
    )

    def _frac_role(sub: pd.DataFrame, role: str) -> float:
        if len(sub) == 0:
            return 0.0
        return float((sub["redox_role"] == role).mean())

    rows = []
    for path, sub in merged.groupby(path_col):
        N = sub[id_col].nunique()
        direct_mean = float(sub["axis_direct"].mean())
        indirect_mean = float(sub["axis_indirect"].mean())
        network_mean = float(sub["axis_network"].mean())

        rows.append(
            {
                path_col: path,
                "N": N,
                "axis_direct_mean": direct_mean,
                "axis_indirect_mean": indirect_mean,
                "axis_network_mean": network_mean,
                "frac_direct_core": _frac_role(sub, "direct_core"),
                "frac_indirect_responder": _frac_role(sub, "indirect_responder"),
                "frac_network_mediator": _frac_role(sub, "network_mediator"),
                "frac_peripheral": _frac_role(sub, "peripheral"),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                path_col,
                "N",
                "axis_direct_mean",
                "axis_indirect_mean",
                "axis_network_mean",
                "frac_direct_core",
                "frac_indirect_responder",
                "frac_network_mediator",
                "frac_peripheral",
            ]
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("axis_direct_mean", ascending=False)
    return out.reset_index(drop=True)
