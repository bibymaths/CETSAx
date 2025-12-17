#!/usr/bin/env python
"""
Script to build sequence-supervised table and train NADPH sequence model.

This script assumes your module (cetsax) provides:
- build_sequence_supervised_table(...)
- NADPHSeqConfig
- train_seq_model(...) -> (model, metrics, cache_paths)

Module stays a pure module (no CLI inside it).
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
import json

import pandas as pd
import torch

from cetsax import (
    build_sequence_supervised_table,
    NADPHSeqConfig,
    train_seq_model,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Build seq-supervised table and train NADPH seq model.")

    # Inputs/outputs
    p.add_argument("--fits-csv", default="../results/ec50_fits.csv", help="Path to ec50_fits.csv")
    p.add_argument("--fasta", default="../results/protein_sequences.fasta",
                   help="Path to FASTA with IDs in headers (>P12345 ... or >sp|P12345|...).")
    p.add_argument("--out-supervised", default="../results/nadph_seq_supervised.csv",
                   help="Output CSV with merged EC50 + sequences + labels.")
    p.add_argument("--out-head", default="../results/nadph_seq_head.pt",
                   help="Output path for trained model.head state_dict.")
    p.add_argument("--out-meta", default="../results/nadph_seq_train_meta.json",
                   help="Output JSON for config/metrics/cache paths.")

    # Task
    p.add_argument("--task", choices=["classification", "regression"], default="classification",
                   help="Train for classification (strong/weak) or regression (-log10 EC50).")
    p.add_argument("--use-nss", action="store_true", default=False,
                   help="If set, use NSS column for regression label_reg (requires NSS in fits table).")

    # Training
    p.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    p.add_argument("--device", default="cuda", help="Device: cuda or cpu.")
    p.add_argument("--batch-size", type=int, default=8,
                   help="ESM token-mode batch size (also used as base for caching batches).")
    p.add_argument("--head-batch-size", type=int, default=256,
                   help="Batch size for pooled/reps head-only training.")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    p.add_argument("--patience", dest="patience", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable/disable early stopping.")

    # Caching / mode
    p.add_argument("--train-mode", choices=["pooled", "reps", "tokens"], default="pooled",
                   help="Training mode: pooled (fast), reps (fast-ish + explain), tokens (slow).")
    p.add_argument("--cache-dir", default="../results/cache_seq_nadph",
                   help="Directory to store caches (tokens/pooled/reps).")
    p.add_argument("--cache-fp16", dest="cache_fp16", action=argparse.BooleanOptionalAction, default=True,
                   help="Store caches in float16 where applicable.")
    p.add_argument("--cache-reps", action="store_true", default=False,
                   help="Also cache residue reps+mask (big, enables explainability without ESM).")

    # --- Model / ESM control ---
    p.add_argument(
        "--model-name",
        default="esm2_t33_650M_UR50D",
        help="ESM model name (must match training/inference)."
    )

    p.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes for classification."
    )

    p.add_argument(
        "--max-len",
        type=int,
        default=1022,
        help="Maximum sequence length (longer sequences are truncated)."
    )

    # --- ESM execution control (IMPORTANT for 1x15GB GPU) ---
    p.add_argument(
        "--esm-batch-size",
        type=int,
        default=2,
        help="Batch size for ESM forward pass / embedding cache (GPU-safe)."
    )

    # --- Cache toggles (explicit, Snakemake-friendly) ---
    p.add_argument(
        "--cache-tokens",
        dest="cache_tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable token cache."
    )

    p.add_argument(
        "--cache-pooled",
        dest="cache_pooled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable pooled embedding cache."
    )

    p.add_argument(
        "--cache-reps",
        dest="cache_reps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable residue-level representation cache."
    )

    args = p.parse_args()

    out_supervised = Path(args.out_supervised)
    out_supervised.parent.mkdir(parents=True, exist_ok=True)

    # 1) Build supervised table
    fits = pd.read_csv(args.fits_csv)

    sup_df = build_sequence_supervised_table(
        fits_df=fits,
        fasta_path=args.fasta,
        out_csv=out_supervised,
        id_col="id",
        use_nss=bool(args.use_nss),
    )
    print(f"Saved supervised table to {out_supervised} (n={len(sup_df)})")

    # 2) Train sequence model (via module)
    cfg = NADPHSeqConfig(
        model_name=args.model_name,
        task=args.task,
        num_classes=args.num_classes,
        epochs=args.epochs,
        device=args.device,

        # batching
        batch_size=args.esm_batch_size,  # ESM / cache batch
        head_batch_size=args.head_batch_size,

        lr=args.lr,
        max_len=args.max_len,

        # training mode
        train_mode=args.train_mode,

        # caching
        cache_dir=args.cache_dir,
        cache_fp16=bool(args.cache_fp16),
        cache_tokens=bool(args.cache_tokens),
        cache_pooled=bool(args.cache_pooled),
        cache_reps=bool(args.cache_reps),
    )

    model, metrics, cache_paths = train_seq_model(
        csv_path=out_supervised,
        cfg=cfg,
        patience=bool(args.patience),
    )

    # Print key result
    if "best_val_acc" in metrics:
        print(f"Training finished. best_val_loss={metrics['best_val_loss']:.6f} best_val_acc={metrics['best_val_acc']:.6f}")
    else:
        print(f"Training finished. best_val_loss={metrics['best_val_loss']:.6f}")

    # 3) Save head
    out_head = Path(args.out_head)
    out_head.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.head.state_dict(), out_head)
    print(f"Saved model head state_dict to {out_head}")

    # 4) Save metadata for reproducibility
    out_meta = Path(args.out_meta)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "config": {
            "task": cfg.task,
            "num_classes": cfg.num_classes,
            "epochs": cfg.epochs,
            "device": cfg.device,
            "batch_size": cfg.batch_size,
            "head_batch_size": getattr(cfg, "head_batch_size", None),
            "lr": cfg.lr,
            "train_mode": cfg.train_mode,
            "cache_dir": cfg.cache_dir,
            "cache_fp16": cfg.cache_fp16,
            "cache_tokens": cfg.cache_tokens,
            "cache_pooled": cfg.cache_pooled,
            "cache_reps": cfg.cache_reps,
            "model_name": cfg.model_name,
            "repr_layer": cfg.repr_layer,
            "max_len": cfg.max_len,
        },
        "metrics": metrics,
        "cache_paths": {k: str(v) for k, v in (cache_paths or {}).items()},
        "artifacts": {
            "supervised_csv": str(out_supervised),
            "head_state_dict": str(out_head),
        },
    }

    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"Saved training metadata to {out_meta}")


if __name__ == "__main__":
    main()