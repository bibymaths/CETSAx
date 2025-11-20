#!/usr/bin/env python
"""
02_hit_calling_and_diagnostics.py

Run hit calling and generate diagnostic plots from EC50 fits.

Inputs:
    - ec50_fits.csv (output of 01_fit_itdr_curves.py)

Outputs:
    - cetsa_hits_ranked.csv
    - ec50_vs_delta_max.png
    - ec50_r1_vs_r2.png (if both replicates present)
    - r2_vs_delta_max.png
    - ec50_vs_r2.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cetsax import run_hit_calling_and_plots


def main() -> None:
    p = argparse.ArgumentParser(description="Hit calling + QC plots for CETSA fits.")
    p.add_argument(
        "--fits-csv",
        # required=True,
        default="../results/ec50_fits.csv",
        help="Path to ec50_fits.csv produced by 01_fit_itdr_curves.py",
    )
    p.add_argument(
        "--out-dir",
        default="../results/hit_results",
        help="Output directory for plots and hit table.",
    )
    args = p.parse_args()

    fits = pd.read_csv(args.fits_csv)
    out_dir = Path(args.out_dir)

    paths = run_hit_calling_and_plots(
        fits_df=fits,
        out_dir=out_dir,
        id_col="id",
        cond_col="condition",
    )

    print(f"Hit calling completed. Ranked hits table at: {out_dir / 'cetsa_hits_ranked.csv'}")
    print("Generated artifacts:")
    for k, v in paths.items():
        if k == "hits_table":
            print(f"  {k}: DataFrame (n={len(v)})")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
