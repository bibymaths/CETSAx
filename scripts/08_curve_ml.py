#!/usr/bin/env python
"""
Script to perform machine learning analyses on CETSA melting curves.
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
from pathlib import Path
from cetsax import load_cetsa_csv, apply_basic_qc
from cetsax import extract_curve_features, classify_curves_kmeans, detect_outliers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-clusters", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_cetsa_csv(args.input_csv)
    df = apply_basic_qc(df)

    # 1. Feature Extraction
    print("Extracting curve shape features...")
    features = extract_curve_features(df, n_components=3)
    features.to_csv(out_dir / "curve_pca_features.csv")

    # 2. Clustering
    print(f"Clustering curves into {args.n_clusters} phenotypes...")
    clusters = classify_curves_kmeans(features, k=args.n_clusters)
    clusters.to_csv(out_dir / "curve_clusters.csv")

    # 3. Outlier Detection
    print("Detecting outliers...")
    outliers = detect_outliers(features)
    outliers.to_csv(out_dir / "curve_outliers.csv")

    n_out = outliers['outlier'].sum()
    print(f"Found {n_out} outlier curves.")


if __name__ == "__main__":
    main()
