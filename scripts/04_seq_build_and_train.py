#!/usr/bin/env python
"""
Script to build sequence-supervised table and train NADPH sequence model.
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
import pandas as pd
import torch

from cetsax import (
    build_sequence_supervised_table,
    NADPHSeqConfig,
    train_seq_model,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Build seq-supervised table and train NADPH seq model.")
    p.add_argument(
        "--fits-csv",
        # required=True,
        default="../results/ec50_fits.csv",
        help="Path to ec50_fits.csv")
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
        help="Train for classification (strong/weak) or regression (e.g. -log10 EC50).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Device to use: 'cuda' or 'cpu'.",
    )

    args = p.parse_args()

    fits = pd.read_csv(args.fits_csv)

    # 1) Build supervised table
    sup_df = build_sequence_supervised_table(
        fits_df=fits,
        fasta_path=args.fasta,
        out_csv=args.out_supervised,
        id_col="id",
        use_nss=True,  # change to True if you later include NSS in the table
    )
    print(f"Saved supervised table with sequences to {args.out_supervised} (n={len(sup_df)})")

    # 2) Train sequence model
    cfg = NADPHSeqConfig(
        task=args.task,
        num_classes=2,
        epochs=args.epochs,
        device=args.device,
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
