#!/usr/bin/env python
"""
08_curve_ml.py
--------------
Unsupervised ML on dose-response curve shapes.
1. Extracts curve features (PCA of dose vectors).
2. Clusters curves (KMeans) to find phenotypes.
3. Detects statistical outliers (noisy/weird curves).
"""
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