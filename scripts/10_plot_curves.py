#!/usr/bin/env python
"""
Script to plot individual protein CETSA curves and global goodness-of-fit.
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
