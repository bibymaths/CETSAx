#!/usr/bin/env python3
"""
Fit EC50 / KD surrogate curves for all proteins in a CETSA NADPH ITDR dataset.
"""

import argparse
from pathlib import Path

from cetsax import load_cetsa_csv, apply_basic_qc, fit_all_proteins


def main() -> None:
    p = argparse.ArgumentParser(description="Fit ITDR EC50 curves for CETSA data.")
    p.add_argument(
        "-i", "--input-csv",
        default="../data/nadph.csv",
        help="Path to CETSA NADPH ITDR CSV file."
    )
    p.add_argument(
        "-o",
        "--output",
        help="Output CSV for fitted parameters.",
        default="ec50_fits.csv",
    )
    args = p.parse_args()

    df = load_cetsa_csv(args.input_csv)
    qc_df = apply_basic_qc(df)
    fit_df = fit_all_proteins(qc_df)

    out_path = Path(args.output)
    fit_df.to_csv(out_path, index=False)
    print(f"Saved fits to {out_path}")


if __name__ == "__main__":
    main()
