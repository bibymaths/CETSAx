#!/usr/bin/env python
"""
Script to predict NADPH responsiveness from protein sequences using a trained ESM + head model.
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
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
import pandas as pd

from cetsax.deeplearn.seq_nadph import (
    read_fasta_to_dict,
    load_esm_model_and_alphabet,
    NADPHSeqModel,
    compute_residue_saliency,
    compute_residue_integrated_gradients,
)


def chunk_list(xs: List, size: int) -> List[List]:
    """Simple chunking utility."""
    return [xs[i: i + size] for i in range(0, len(xs), size)]


def predict_nadph_from_fasta(
        fasta_path: str | Path,
        checkpoint: str | Path,
        model_name: str = "esm2_t33_650M_UR50D",
        task: str = "classification",
        num_classes: int = 3,
        batch_size: int = 8,
        max_len: int = 1022,
        device: str | None = None,
        compute_saliency: bool = False,
        compute_ig: bool = False,
        target_class: int | None = None,
        ig_steps: int = 50,
) -> pd.DataFrame:
    fasta_path = Path(fasta_path)
    checkpoint = Path(checkpoint)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    print(f"Using device: {device}")  # <--- [MONITORING 1] Device check

    # 1) Load sequences
    seq_dict: Dict[str, str] = read_fasta_to_dict(fasta_path)
    ids = list(seq_dict.keys())
    seqs = [seq_dict[i] for i in ids]

    if len(ids) == 0:
        raise ValueError(f"No sequences found in FASTA: {fasta_path}")

    # <--- [MONITORING 2] Data Stats
    print(f"Loaded {len(ids)} sequences from {fasta_path.name}")

    # 2) Load ESM backbone + alphabet
    print(f"Loading ESM model: {model_name}...")  # <--- [MONITORING 3] Model Status
    esm_model, alphabet = load_esm_model_and_alphabet(model_name)
    esm_model.to(device_t)
    batch_converter = alphabet.get_batch_converter()

    # 3) Build NADPHSeqModel and load head weights
    embed_dim = esm_model.embed_dim
    model = NADPHSeqModel(
        esm_model,
        embed_dim=embed_dim,
        task=task,
        num_classes=num_classes,
    ).to(device_t)

    # load head-only checkpoint
    print(f"Loading checkpoint: {checkpoint.name}...")  # <--- [MONITORING]
    state = torch.load(checkpoint, map_location=device_t)
    model.head.load_state_dict(state)
    model.eval()

    # 4) Inference in batches
    all_rows = []
    int_to_class = {0: "weak", 1: "strong"}
    grad_context = torch.no_grad if not (compute_saliency or compute_ig) else lambda: torch.enable_grad()

    # <--- [MONITORING 4] Batch calculation
    chunks = chunk_list(ids, batch_size)
    total_batches = len(chunks)
    print(f"Starting inference: {total_batches} batches to process.")

    with grad_context():
        # <--- [MONITORING 5] Enumerate to track index
        for batch_idx, id_chunk in enumerate(chunks):

            # Print progress every 10 batches (or every batch if you prefer)
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(f"  Processing batch {batch_idx + 1}/{total_batches} ({(batch_idx + 1) / total_batches:.1%})")

            seq_chunk = [seq_dict[i] for i in id_chunk]

            # Optionally truncate sequences
            seq_chunk_trunc: List[Tuple[str, str]] = []
            for pid, seq in zip(id_chunk, seq_chunk):
                if len(seq) > max_len:
                    seq = seq[:max_len]
                seq_chunk_trunc.append((pid, seq))

            # ESM batch conversion
            _, _, toks = batch_converter(seq_chunk_trunc)
            toks = toks.to(device_t)

            # --- forward pass
            outputs = model(toks)

            # --- base predictions
            if task == "classification":
                probs = F.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)
                probs_np = probs.detach().cpu().numpy()
                preds_np = preds.detach().cpu().numpy()
            else:
                preds = outputs
                preds_np = preds.detach().cpu().numpy()

            # --- interpretability (optional)
            saliency_scores = None
            ig_scores = None

            if compute_saliency:
                saliency_scores = compute_residue_saliency(
                    model, toks, target_class=target_class,
                ).detach()

            if compute_ig:
                ig_scores = compute_residue_integrated_gradients(
                    model, toks, target_class=target_class, steps=ig_steps,
                ).detach()

                # --- assemble per-protein rows
                for i, pid in enumerate(id_chunk):
                    row = {"id": pid}

                    # Get the actual sequence length for this protein
                    # (We use seq_chunk_trunc because that's what went into the model)
                    # ESM-2 structure: [CLS, res1, res2, ..., resN, EOS, PAD, PAD...]
                    # Indices:          0     1     2          N    N+1
                    seq_len = len(seq_chunk_trunc[i][1])  # [1] is the sequence string

                    # Prediction Logic (unchanged)
                    if task == "classification":
                        pc = int(preds_np[i])
                        row["pred_class_idx"] = pc
                        row["pred_class"] = int_to_class.get(pc, f"class_{pc}")
                        for k in range(probs_np.shape[1]):
                            row[f"p_class{k}"] = float(probs_np[i, k])
                    else:
                        row["pred_value"] = float(preds_np[i])

                    # We slice from 1 to seq_len + 1 to skip <cls> and <eos>

                    if compute_saliency and saliency_scores is not None:
                        # Slice exactly the residues
                        sal = saliency_scores[i, 1: seq_len + 1].cpu().numpy()
                        row["saliency"] = ";".join(f"{x:.6f}" for x in sal)

                    if compute_ig and ig_scores is not None:
                        # Slice exactly the residues
                        ig = ig_scores[i, 1: seq_len + 1].cpu().numpy()
                        row["ig"] = ";".join(f"{x:.6f}" for x in ig)

                    all_rows.append(row)

    return pd.DataFrame(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict NADPH responsiveness from protein sequence using ESM + trained head."
    )
    parser.add_argument(
        "--fasta",
        # required=True,
        default="../results/protein_sequences.fasta",
        help="Path to FASTA file with protein sequences (headers must contain IDs).",
    )
    parser.add_argument(
        "--checkpoint",
        # required=True,
        default="../results/nadph_seq_head.pt",
        help="Path to trained model.head state_dict (e.g., nadph_seq_head.pt).",
    )
    parser.add_argument(
        "--model-name",
        default="esm2_t33_650M_UR50D",
        help="ESM model name used during training.",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
        help="Prediction task type.",
    )
    parser.add_argument(
        "--saliency",
        action="store_true",
        help="If set, compute per-residue saliency scores.",
    )
    parser.add_argument(
        "--ig",
        action="store_true",
        help="If set, compute per-residue integrated gradients.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Target class index for saliency/IG (classification only). "
             "If omitted, uses predicted class per sample.",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=50,
        help="Number of interpolation steps for integrated gradients.",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes for classification.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1022,
        help="Maximum sequence length (truncate longer sequences).",
    )
    parser.add_argument(
        "--device",
        default='cpu',
        help="Device to use: 'cuda', 'cpu', or leave empty for auto.",
    )
    parser.add_argument(
        "--out",
        # required=True,
        default="../results/predictions_nadph_seq.csv",
        help="Output CSV path for predictions.",
    )

    args = parser.parse_args()

    df_pred = predict_nadph_from_fasta(
        fasta_path=args.fasta,
        checkpoint=args.checkpoint,
        model_name=args.model_name,
        task=args.task,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        max_len=args.max_len,
        device=args.device,
        compute_saliency=args.saliency,
        compute_ig=args.ig,
        target_class=args.target_class,
        ig_steps=args.ig_steps,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
