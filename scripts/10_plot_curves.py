#!/usr/bin/env python
"""
10_plot_curves.py
-----------------
Generates individual curve plots for significant hits.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from cetsax import load_cetsa_csv, apply_basic_qc
from cetsax import plot_protein_curve, plot_goodness_of_fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--fits-csv", required=True)
    parser.add_argument("--hits-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    df = load_cetsa_csv(args.input_csv)
    df = apply_basic_qc(df)
    fits = pd.read_csv(args.fits_csv)
    hits = pd.read_csv(args.hits_csv)

    # 1. Global Goodness of Fit
    print("Plotting global goodness of fit...")
    fig, ax = plot_goodness_of_fit(df, fits)
    fig.savefig(out_dir / "global_goodness_of_fit.png")
    plt.close(fig)

    # 2. Individual Hit Curves
    # Plot Top 50 Strong/Medium hits
    top_hits = hits[hits['dominant_class'].isin(['strong', 'medium'])].head(50)

    curves_dir = out_dir / "individual_curves"
    curves_dir.mkdir(exist_ok=True)

    print(f"Plotting {len(top_hits)} individual curves...")
    for _, row in top_hits.iterrows():
        prot_id = row['id']
        try:
            # Plot per condition if multiple exist
            # Assuming fits file has 'condition' column
            conditions = fits[fits['id'] == prot_id]['condition'].unique()

            for cond in conditions:
                fig, ax = plt.subplots()
                plot_protein_curve(df, fits, prot_id, condition=cond, ax=ax)
                fig.savefig(curves_dir / f"{prot_id}_{cond}.png")
                plt.close(fig)
        except Exception as e:
            print(f"Skipped {prot_id}: {e}")


if __name__ == "__main__":
    main()
