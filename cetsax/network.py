"""
network.py
----------

Purpose:
    Construct proteome-wide co-stabilization matrices and derive
    CETSA-based thermal stability networks.

    This module supports:
    - Pairwise co-stabilization correlation matrices
    - Condition-specific or replicate-specific networks
    - Adjacency matrix thresholding for graph construction
    - Graph-based clustering (community detection, modules)
    - Extraction of NADPH-responsive protein modules

    These networks allow identification of:
    - direct/indirect binder communities
    - metabolic or structural complexes responding together
    - dose-sensitive co-stabilization signatures
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from typing import Optional, Tuple, Dict

from .config import DOSE_COLS, ID_COL


def compute_costab_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute co-stabilization (correlation) matrix across proteins.
    Each protein is represented by its dose-response vector.

    Returns:
        DataFrame (n_proteins x n_proteins) correlation matrix.
    """
    dose_mat = df.groupby(ID_COL)[DOSE_COLS].mean()  # average replicates
    corr = dose_mat.T.corr(method="pearson")
    return corr


def make_network_from_matrix(
    corr_matrix: pd.DataFrame,
    cutoff: float = 0.7
) -> nx.Graph:
    """
    Convert correlation matrix into a network graph.
    Edges exist for correlations >= cutoff.
    """
    G = nx.Graph()
    proteins = corr_matrix.index

    for i, p1 in enumerate(proteins):
        for j, p2 in enumerate(proteins):
            if j <= i:
                continue
            r = corr_matrix.iloc[i, j]
            if np.abs(r) >= cutoff:
                G.add_edge(p1, p2, weight=r)

    return G


def detect_modules(G: nx.Graph) -> Dict[str, int]:
    """
    Apply a community detection algorithm (Louvain or greedy modularity)
    to identify co-stabilization modules.

    Returns:
        dict: {protein_id: module_id}
    """
    from networkx.algorithms.community import greedy_modularity_communities

    comms = greedy_modularity_communities(G)
    module_map = {}
    for i, comm in enumerate(comms):
        for node in comm:
            module_map[node] = i
    return module_map
