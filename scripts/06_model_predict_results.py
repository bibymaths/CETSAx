#!/usr/bin/env python
import argparse
from cetsax import visualize_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-file", required=True)
    parser.add_argument("--truth-file", required=True)
    args = parser.parse_args()

    visualize_predictions(
        pred_file=args.pred_file,
        truth_file=args.truth_file
    )