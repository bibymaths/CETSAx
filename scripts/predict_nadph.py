#!/usr/bin/env python
"""
predict_nadph_from_seq.py
-------------------------

Use a trained sequence-based NADPH model to predict responsiveness
for new protein sequences.

Assumptions:
    - You trained using cetsax.deeplearn.seq_nadph.train_seq_model
    - After training, you saved the head weights via:

        torch.save(model.head.state_dict(), "nadph_seq_head.pt")

Usage (classification example):

    python predict_nadph_from_seq.py \
        --fasta protein_sequences.fasta \
        --checkpoint nadph_seq_head.pt \
        --task classification \
        --num-classes 3 \
        --out predictions_nadph_seq.csv
"""

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
)


def chunk_list(xs: List, size: int) -> List[List]:
    """Simple chunking utility."""
    return [xs[i : i + size] for i in range(0, len(xs), size)]


def predict_nadph_from_fasta(
    fasta_path: str | Path,
    checkpoint: str | Path,
    model_name: str = "esm2_t33_650M_UR50D",
    task: str = "classification",  # or "regression"
    num_classes: int = 3,
    batch_size: int = 8,
    max_len: int = 1022,
    device: str | None = None,
) -> pd.DataFrame:
    """
    Run inference on sequences in FASTA.

    Parameters
    ----------
    fasta_path : str or Path
        Input FASTA (headers must have protein IDs: >P12345 ...).

    checkpoint : str or Path
        Path to saved model.head state_dict (see docstring above).

    model_name : str
        ESM-2 model name, must match training (default: esm2_t33_650M_UR50D).

    task : {'classification', 'regression'}
        Prediction type:
            - classification: strong/medium/weak
            - regression: continuous target (e.g. -log10 EC50)

    num_classes : int
        Number of classes (for classification). Default: 3.

    batch_size : int
        Inference batch size.

    max_len : int
        Maximum sequence length to use (truncate longer sequences).

    device : str or None
        'cuda', 'cpu', or None to auto-select.

    Returns
    -------
    DataFrame
        For classification:
            id, pred_class, p_class0, p_class1, p_class2
        For regression:
            id, pred_value
    """
    fasta_path = Path(fasta_path)
    checkpoint = Path(checkpoint)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    # 1) Load sequences
    seq_dict: Dict[str, str] = read_fasta_to_dict(fasta_path)
    ids = list(seq_dict.keys())
    seqs = [seq_dict[i] for i in ids]

    if len(ids) == 0:
        raise ValueError(f"No sequences found in FASTA: {fasta_path}")

    # 2) Load ESM backbone + alphabet
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
    state = torch.load(checkpoint, map_location=device_t)
    model.head.load_state_dict(state)
    model.eval()

    # 4) Inference in batches
    all_rows = []

    # Mapping from class index to label (you can adapt if you change your labels)
    int_to_class = {0: "weak", 1: "medium", 2: "strong"}

    with torch.no_grad():
        for id_chunk in chunk_list(ids, batch_size):
            seq_chunk = [seq_dict[i] for i in id_chunk]

            # Optionally truncate sequences
            seq_chunk_trunc: List[Tuple[str, str]] = []
            for pid, seq in zip(id_chunk, seq_chunk):
                if len(seq) > max_len:
                    seq = seq[:max_len]
                seq_chunk_trunc.append((pid, seq))

            # ESM batch conversion: returns (labels, strs, toks)
            _, _, toks = batch_converter(seq_chunk_trunc)
            toks = toks.to(device_t)

            outputs = model(toks)

            if task == "classification":
                probs = F.softmax(outputs, dim=1)  # (B, num_classes)
                preds = probs.argmax(dim=1)        # (B,)

                probs_np = probs.cpu().numpy()
                preds_np = preds.cpu().numpy()

                for pid, pvec, pc in zip(id_chunk, probs_np, preds_np):
                    row = {
                        "id": pid,
                        "pred_class_idx": int(pc),
                        "pred_class": int_to_class.get(int(pc), f"class_{pc}"),
                    }
                    # add per-class probabilities
                    for k in range(probs_np.shape[1]):
                        row[f"p_class{k}"] = float(pvec[k])
                    all_rows.append(row)

            else:  # regression
                preds = outputs  # (B,)
                preds_np = preds.cpu().numpy()
                for pid, val in zip(id_chunk, preds_np):
                    all_rows.append(
                        {
                            "id": pid,
                            "pred_value": float(val),
                        }
                    )

    return pd.DataFrame(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict NADPH responsiveness from protein sequence using ESM + trained head."
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Path to FASTA file with protein sequences (headers must contain IDs).",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
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
        "--num-classes",
        type=int,
        default=3,
        help="Number of classes for classification.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
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
        default=None,
        help="Device to use: 'cuda', 'cpu', or leave empty for auto.",
    )
    parser.add_argument(
        "--out",
        required=True,
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
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
