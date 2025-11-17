#!/usr/bin/env python3
"""
Call CETSA binding hits from EC50 fits and produce a protein-level summary.
"""

import argparse
from pathlib import Path

import pandas as pd

from cetsax import call_hits, summarize_hits


def main() -> None:
    p = argparse.ArgumentParser(description="Call CETSA ITDR binding hits from fitted EC50 table.")
    p.add_argument("fit_csv", help="CSV produced by fit_ec50.py")
    p.add_argument(
        "-o",
        "--output_hits",
        default="ec50_hits.csv",
        help="Output CSV for per-replicate hits.",
    )
    p.add_argument(
        "-s",
        "--output_summary",
        default="ec50_hits_summary.csv",
        help="Output CSV for per-protein hit summary.",
    )
    p.add_argument("--r2-min", type=float, default=0.8, help="Minimum R^2 to accept a fit.")
    p.add_argument(
        "--delta-min",
        type=float,
        default=0.1,
        help="Minimum max signal change across doses to call a hit.",
    )
    p.add_argument(
        "--min-reps",
        type=int,
        default=2,
        help="Minimum number of replicates where a protein must be a hit.",
    )
    args = p.parse_args()

    fit_df = pd.read_csv(args.fit_csv)
    hits_df = call_hits(fit_df, r2_min=args.r2_min, delta_min=args.delta_min)
    summary_df = summarize_hits(hits_df, min_reps=args.min_reps)

    hits_path = Path(args.output_hits)
    summary_path = Path(args.output_summary)

    hits_df.to_csv(hits_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved per-replicate hits to {hits_path}")
    print(f"Saved per-protein hit summary to {summary_path}")


if __name__ == "__main__":
    main()
