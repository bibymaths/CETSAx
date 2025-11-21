"""
network.py
----------
Network construction and module detection based on co-stabilization.
This module provides functions to compute co-stabilization (correlation)
matrices from CETSA dose-response data, build protein-protein interaction
networks based on correlation cutoffs, and detect co-stabilization modules
using community detection algorithms.
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

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict
from networkx.algorithms.community import greedy_modularity_communities
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
    comms = greedy_modularity_communities(G)
    module_map = {}
    for i, comm in enumerate(comms):
        for node in comm:
            module_map[node] = i
    return module_map
