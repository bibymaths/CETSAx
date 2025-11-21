#!/usr/bin/env python
"""
09_bayesian_fit.py
------------------
Performs Bayesian inference on the Top N hits to estimate
posterior distributions for EC50 (uncertainty quantification).

Parallelized using joblib to utilize all available CPU cores.
"""
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional

# Parallelization imports
from joblib import Parallel, delayed
from tqdm import tqdm

# Import core logic from the package
from cetsax import load_cetsa_csv, apply_basic_qc, bayesian_fit_ec50


def _fit_single_protein(df: pd.DataFrame, prot_id: str) -> Optional[pd.DataFrame]:
    """
    Worker function to fit a single protein.

    Wraps the bayesian_fit_ec50 call in a try/except block for safety
    during parallel execution. Returns the summary dataframe or None.
    """
    try:
        # Run the MCMC sampling
        # Note: Since joblib creates separate processes, PyMC will run
        # independently in each process.
        res = bayesian_fit_ec50(df, prot_id)

        summary = res['summary']
        summary['id'] = prot_id
        return summary

    except Exception as e:
        # Print error but don't crash the whole pipeline
        print(f"  [Warning] Failed fit for {prot_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run Bayesian EC50 inference in parallel.")
    parser.add_argument("--input-csv", required=True, help="Path to raw CETSA data")
    parser.add_argument("--hits-csv", required=True, help="Path to ranked hits file")
    parser.add_argument("--out-dir", required=True, help="Directory to save results")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top hits to analyze")

    # New argument for core control
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs (-1 for all cores)")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and clean data
    # This is done once in the main process, then shared with workers via copy-on-write
    print(f"Loading data from {args.input_csv}...")
    df = load_cetsa_csv(args.input_csv)
    df = apply_basic_qc(df)

    # 2. Select targets
    hits = pd.read_csv(args.hits_csv)
    top_targets = hits.head(args.top_n)['id'].tolist()

    print(f"Running Bayesian inference for top {len(top_targets)} targets using {args.n_jobs} cores...")

    # 3. Run Parallel Inference
    # Parallel() creates the pool. delayed() wraps the function call.
    # tqdm() provides the progress bar.
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(_fit_single_protein)(df, pid)
        for pid in tqdm(top_targets, desc="Bayesian MCMC Sampling")
    )

    # 4. Aggregate results
    # Filter out None values (failed fits)
    summaries = [res for res in results if res is not None]

    if summaries:
        final_df = pd.concat(summaries)

        # Move 'id' column to the front for better readability
        cols = ['id'] + [c for c in final_df.columns if c != 'id']
        final_df = final_df[cols]

        out_file = out_dir / "bayesian_ec50_summaries.csv"
        final_df.to_csv(out_file)
        print(f"Successfully saved Bayesian summaries for {len(summaries)} proteins to:")
        print(f"  {out_file}")
    else:
        print("No successful fits were generated.")


if __name__ == "__main__":
    main()
