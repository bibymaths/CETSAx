#!/usr/bin/env python
"""
04_seq_build_and_train.py

Build sequence-based supervised table and train NADPH responsiveness model
(ESM-2 + MLP head) using cetsax.deeplearn.seq_nadph.

Inputs:
    - ec50_fits.csv
    - protein_sequences.fasta (headers like >P12345 ...)

Outputs:
    - nadph_seq_supervised.csv
    - nadph_seq_head.pt (head-only state dict for inference)
"""

from __future__ import annotations

import argparse
import pandas as pd
import torch

from cetsax import (
    build_sequence_supervised_table,
    NADPHSeqConfig,
    train_seq_model,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Build seq-supervised table and train NADPH seq model.")
    p.add_argument("--fits-csv", required=True, help="Path to ec50_fits.csv")
    p.add_argument(
        "--fasta",
        # required=True,
        default="../results/protein_sequences.fasta",
        help="Path to protein_sequences.fasta with IDs in headers (>P12345 ...).",
    )
    p.add_argument(
        "--out-supervised",
        default="../results/nadph_seq_supervised.csv",
        help="Output CSV with merged EC50 + sequences + labels.",
    )
    p.add_argument(
        "--out-head",
        default="../results/nadph_seq_head.pt",
        help="Output path for trained model.head state_dict.",
    )
    p.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
        help="Train for classification (strong/medium/weak) or regression (e.g. -log10 EC50).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    args = p.parse_args()

    fits = pd.read_csv(args.fits_csv)

    # 1) Build supervised table
    sup_df = build_sequence_supervised_table(
        fits_df=fits,
        fasta_path=args.fasta,
        out_csv=args.out_supervised,
        id_col="id",
        use_nss=False,  # change to True if you later include NSS in the table
    )
    print(f"Saved supervised table with sequences to {args.out_supervised} (n={len(sup_df)})")

    # 2) Train sequence model
    cfg = NADPHSeqConfig(
        task=args.task,
        num_classes=3,
        epochs=args.epochs,
    )

    model, metrics = train_seq_model(
        csv_path=args.out_supervised,
        cfg=cfg,
    )
    print("Training finished. Best validation metric:", metrics["best_val"])

    # 3) Save head
    torch.save(model.head.state_dict(), args.out_head)
    print(f"Saved model head state_dict to {args.out_head}")


if __name__ == "__main__":
    main()
