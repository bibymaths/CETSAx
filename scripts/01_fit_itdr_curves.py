#!/usr/bin/env python
"""
01_fit_itdr_curves.py

Run EC50 / ITDR curve fitting using the cetsax package.

Inputs:
    - CETSA NADPH ITDR CSV (normalized, QC'ed)

Outputs:
    - ec50_fits.csv : per (id, condition) fit parameters
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cetsax import ID_COL, COND_COL, DOSE_COLS, load_cetsa_csv, apply_basic_qc
from cetsax import fit_all_proteins


def main() -> None:
    p = argparse.ArgumentParser(description="Fit ITDR EC50 curves for all proteins.")
    p.add_argument(
        "--input-csv",
        # required=True,
        default="../data/nadph.csv",
        help="Path to CETSA NADPH ITDR CSV file.",
    )
    p.add_argument(
        "--out-fits",
        default="../results/ec50_fits.csv",
        help="Output CSV for fitted parameters.",
    )
    args = p.parse_args()

    in_path = Path(args.input_csv)
    out_path = Path(args.out_fits)

    df = load_cetsa_csv(str(in_path))
    qc_df = apply_basic_qc(df)

    # Optional: basic sanity check
    missing = [c for c in [ID_COL, COND_COL] + DOSE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


    fits_df = fit_all_proteins(qc_df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fits_df.to_csv(out_path, index=False)

    print(f"Saved EC50 fits to {out_path} (n={len(fits_df)})")


if __name__ == "__main__":
    main()
