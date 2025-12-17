#!/usr/bin/env python
"""
Script to build sequence-supervised table and train NADPH sequence model.
UPDATED: Compatible with cetsax.deeplearn.seq_nadph (Transformers backend).
"""
# BSD 3-Clause License
# Copyright (c) 2025, Abhinav Mishra

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
    p = argparse.ArgumentParser(description="Build seq-supervised table and train NADPH seq model (HF Backend).")

    # Inputs/outputs
    p.add_argument("--fits-csv", default="results/ec50_fits.csv", help="Path to ec50_fits.csv")
    p.add_argument("--fasta", default="results/protein_sequences.fasta",
                   help="Path to FASTA.")
    p.add_argument("--out-supervised", default="results/nadph_seq_supervised.csv",
                   help="Output CSV with merged EC50 + sequences + labels.")
    p.add_argument("--out-head", default="results/nadph_seq_head.pt",
                   help="Output path for trained model.head state_dict.")
    p.add_argument("--out-meta", default="results/nadph_seq_train_meta.json",
                   help="Output JSON for config/metrics/cache paths.")
    p.add_argument("--out-info", default="results/nadph_seq_train_info.csv",
                   help="Output CSV for per-epoch training info.")

    # Task
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--use-nss", action="store_true", default=False)

    # Training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", dest="patience", action=argparse.BooleanOptionalAction, default=True)

    # Batching & Accumulation (New for 15GB GPU optimization)
    p.add_argument("--batch-size", type=int, default=2,
                   help="Actual GPU batch size (keep small for ESM-650M).")
    p.add_argument("--accum-steps", type=int, default=4,
                   help="Gradient accumulation steps (Simulated Batch Size = batch_size * accum_steps).")
    p.add_argument("--head-batch-size", type=int, default=256,
                   help="Batch size when training head-only (pooled/reps mode).")

    # Fine-tuning
    p.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True,
                   help="If False, fine-tunes the ESM backbone (only in 'tokens' mode).")

    # Caching / mode
    p.add_argument("--train-mode", choices=["pooled", "reps", "tokens"], default="pooled",
                   help="pooled=Fastest, reps=Explainable, tokens=Fine-tuning.")
    p.add_argument("--cache-dir", default="results/cache_seq_nadph")
    p.add_argument("--cache-fp16", dest="cache_fp16", action=argparse.BooleanOptionalAction, default=True)

    # Model (HF format)
    p.add_argument(
        "--model-name",
        default="facebook/esm2_t33_650M_UR50D",
        help="Hugging Face model ID."
    )
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--max-len", type=int, default=1022)

    # --- Cache toggles ---
    p.add_argument(
        "--use-token-cache",
        dest="cache_tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable token cache.",
    )

    p.add_argument(
        "--use-pooled-cache",
        dest="cache_pooled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable pooled embedding cache.",
    )

    p.add_argument(
        "--use-reps-cache",
        dest="cache_reps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable/disable residue-level reps cache (large).",
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

    # 2) Train sequence model
    cfg = NADPHSeqConfig(
        model_name=args.model_name,
        task=args.task,
        num_classes=args.num_classes,
        epochs=args.epochs,
        device=args.device,

        # New Batching Logic
        batch_size=args.batch_size,  # GPU BS
        accum_steps=args.accum_steps,  # Grad Accumulation
        head_batch_size=args.head_batch_size,

        lr=args.lr,
        max_len=args.max_len,
        freeze_backbone=args.freeze_backbone,

        train_mode=args.train_mode,
        cache_dir=args.cache_dir,
        cache_fp16=bool(args.cache_fp16),
        cache_tokens=bool(args.cache_tokens),
        cache_pooled=bool(args.cache_pooled),
        cache_reps=bool(args.cache_reps),
    )

    model, metrics, cache_paths, run_info = train_seq_model(
        csv_path=out_supervised,
        cfg=cfg,
        patience=bool(args.patience),
    )

    if "best_val_acc" in metrics:
        print(f"Finished. Best Val Loss={metrics['best_val_loss']:.4f}, Acc={metrics['best_val_acc']:.4f}")
    else:
        print(f"Finished. Best Val Loss={metrics['best_val_loss']:.4f}")

    # 3) Save head (or full model if fine-tuned)
    out_head = Path(args.out_head)
    out_head.parent.mkdir(parents=True, exist_ok=True)

    # Logic: If we fine-tuned backbone, we should technically save the whole model or PEFT adapter.
    # For this script, we default to saving the HEAD.
    # If the user fine-tuned ('tokens' + no-freeze), this head checkpoint is useful but insufficient
    # to reproduce results without the fine-tuned backbone.
    # We warn the user if that's the case.
    if args.train_mode == "tokens" and not args.freeze_backbone:
        print("[WARNING] You fine-tuned the backbone. Saving only head.pt is insufficient for full inference.")
        print("[WARNING] Consider modifying script to save full model for fine-tuning scenarios.")

    if hasattr(model, "head"):
        torch.save(model.head.state_dict(), out_head)
    else:
        # Fallback if model structure varied
        torch.save(model.state_dict(), out_head)

    print(f"Saved model head state_dict to {out_head}")

    # 4) Save metadata
    out_meta = Path(args.out_meta)
    meta = {
        "config": vars(args),  # Dumps all args
        "metrics": metrics,
        "cache_paths": {k: str(v) for k, v in (cache_paths or {}).items()},
        "artifacts": {
            "supervised_csv": str(out_supervised),
            "head_state_dict": str(out_head),
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    out_info = Path(args.out_info)
    run_info.to_csv(out_info, index=False)
    print(f"Saved info to {out_info}")


if __name__ == "__main__":
    main()