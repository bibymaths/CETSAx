#!/usr/bin/env python
"""
Script to run Bayesian EC50 inference in parallel using joblib.
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
