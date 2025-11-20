"""
redox.py
--------

Redox-axis reconstruction from NADPH CETSA data.

Purpose
-------
Use fitted CETSA parameters, NADPH sensitivity scores, hit classes,
and (optionally) co-stabilization network centrality to reconstruct
a "redox landscape" with:

    - Direct NADPH cores (high affinity, strong effect, clean curves)
    - Indirect / downstream responders (medium affinity/effect)
    - Network mediators (high centrality, moderate response)
    - Peripheral / non-responders

This module defines three conceptual axes per protein:

    Axis 1: direct_redox_axis
        High when EC50 is low, delta_max high, hit_class strong.

    Axis 2: indirect_redox_axis
        High when EC50 is moderate, delta_max moderate, hit_class medium,
        especially if not strongly direct.

    Axis 3: network_redox_axis
        High when network centrality (degree / betweenness) is high,
        regardless of direct vs indirect.

Typical usage
-------------
1. Start from:
    - fits_df (ec50_fits.csv; per-protein median summaries)
    - sens_df (from sensitivity.compute_sensitivity_scores)
    - hits_df (from ranked hit table; per-protein dominant hit class)
    - net_df  (optional; per-protein network centrality stats)
    - annot_df (id -> pathway)

2. Use:
    - build_redox_axes(...)         -> per-protein redox scores + role
    - summarize_redox_by_pathway(...) -> pathway-level redox patterns
"""

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
