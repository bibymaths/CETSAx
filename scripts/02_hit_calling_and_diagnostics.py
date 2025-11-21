#!/usr/bin/env python
"""
Script to perform hit calling and generate diagnostic plots for CETSA EC50 fits.
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
