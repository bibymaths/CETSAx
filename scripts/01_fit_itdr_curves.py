#!/usr/bin/env python
"""
Script to fit ITDR EC50 curves for all proteins in a CETSA dataset.
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
