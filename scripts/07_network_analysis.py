#!/usr/bin/env python
"""
Script to build CETSA co-stabilization network from processed data.
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

import argparse
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from cetsax import load_cetsa_csv, apply_basic_qc
from cetsax import compute_costab_matrix, make_network_from_matrix, detect_modules


def main():
    parser = argparse.ArgumentParser(description="Build CETSA co-stabilization network")
    parser.add_argument("--input-csv", required=True, help="Raw CETSA data")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--corr-cutoff", type=float, default=0.85, help="Correlation threshold for edges")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and Matrix
    print("Loading data...")
    df = load_cetsa_csv(args.input_csv)
    df = apply_basic_qc(df)

    print("Computing co-stabilization matrix...")
    corr_matrix = compute_costab_matrix(df)
    corr_matrix.to_csv(out_dir / "costab_matrix.csv")

    # Plot heat-map of co-stabilization matrix
    plt.figure(figsize=(16, 16))
    sns.heatmap(corr_matrix, cmap='viridis')
    plt.title('Co-stabilization Matrix Heatmap')
    plt.savefig(out_dir / "costab_matrix_heatmap.png", dpi=300)
    plt.close()

    # 2. Build Network
    print(f"Building graph (cutoff={args.corr_cutoff})...")
    G = make_network_from_matrix(corr_matrix, cutoff=args.corr_cutoff)
    nx.write_gexf(G, out_dir / "network_graph.gexf")
    print(f"Graph saved with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # 3. Detect Modules
    print("Detecting communities...")
    modules = detect_modules(G)
    mod_df = pd.DataFrame(list(modules.items()), columns=['id', 'module_id'])
    mod_df.to_csv(out_dir / "network_modules.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
