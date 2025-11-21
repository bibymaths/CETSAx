#!/usr/bin/env python
import argparse
from cetsax import visualize_predictions, analyze_fitting_data, generate_bio_insight

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-file", required=True)
    parser.add_argument("--pred-file", required=True)
    parser.add_argument("--truth-file", required=True)
    parser.add_argument("--annot-file", required=True)
    args = parser.parse_args()

    visualize_predictions(
        pred_file=args.pred_file,
        truth_file=args.truth_file
    )

    analyze_fitting_data(
        pred_file=args.pred_file,
        fits_file=args.fit_file,
    )

    generate_bio_insight(
        pred_file=args.pred_file,
        annot_file=args.annot_file,
        truth_file=args.truth_file,
    )
