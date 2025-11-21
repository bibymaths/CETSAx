#!/usr/bin/env python
"""
07_network_analysis.py
----------------------
Builds a co-stabilization network from raw CETSA data to identify
protein modules that respond similarly to NADPH.

Outputs:
    - costab_matrix.csv: Pairwise correlations
    - network_modules.csv: Community detection results
    - network_graph.gexf: Graph file for visualization in Gephi/Cytoscape
"""
import argparse
import pandas as pd
import networkx as nx
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
