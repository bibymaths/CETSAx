#!/usr/bin/env python
"""
Predict NADPH responsiveness from protein sequences (Transformers Backend).

Modes:
- pooled: uses cached pooled embeddings (fastest)
- reps:   uses cached residue reps (fast, supports Saliency/IG)
- tokens: runs HF ESM forward (slow, supports Saliency/IG)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


# -----------------------------
# Helpers
# -----------------------------
def chunk_list(xs: List, size: int) -> List[List]:
    return [xs[i:i + size] for i in range(0, len(xs), size)]


def _auto_device(device: str) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_fasta_to_dict(fasta_path: str | Path) -> Dict[str, str]:
    fasta_path = Path(fasta_path)
    seqs: Dict[str, List[str]] = {}
    current_id: Optional[str] = None
    with fasta_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                header_token = line[1:].split()[0]
                current_id = header_token.split("|")[1] if "|" in header_token else header_token
                seqs.setdefault(current_id, [])
            elif current_id:
                seqs[current_id].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


# -----------------------------
# Models (Matching the new module)
# -----------------------------
class AttentionPooling(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        # Matches the definition in new seq_nadph.py
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attention(x).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), -1e9)
        weights = F.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


class HeadOnlyModel(nn.Module):
    """Wrapper for pooled embeddings -> Predictions"""

    def __init__(self, embed_dim: int, task: str, num_classes: int = 2):
        super().__init__()
        self.task = task
        self.norm = nn.LayerNorm(embed_dim)
        hidden = 512  # Matched to new module
        out_dim = num_classes if task == "classification" else 1

        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),  # New module uses GELU
            nn.Dropout(0.4),  # New module uses 0.4
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        logits = self.head(x)
        return logits.squeeze(-1) if self.task == "regression" else logits


class RepsHeadModel(nn.Module):
    """Wrapper for Residue Reps -> Pooling -> Predictions"""

    def __init__(self, embed_dim: int, task: str, num_classes: int = 2):
        super().__init__()
        self.task = task
        self.pooler = AttentionPooling(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        hidden = 512
        out_dim = num_classes if task == "classification" else 1
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, reps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = self.pooler(reps, mask)
        x = self.norm(pooled)
        logits = self.head(x)
        return logits.squeeze(-1) if self.task == "regression" else logits


# -----------------------------
# Cache Loaders
# -----------------------------
def _load_pooled_cache(pt_path):
    obj = torch.load(pt_path, map_location="cpu")
    return obj["pooled"]


def _load_reps_cache(pt_path):
    obj = torch.load(pt_path, map_location="cpu")
    return obj["reps"], obj["mask"]


def _load_supervised_ids(csv_path):
    df = pd.read_csv(csv_path)
    return df["id"].astype(str).tolist()


# -----------------------------
# Main
# -----------------------------
def predict(args):
    device = _auto_device(args.device)
    print(f"Device: {device} | Mode: {args.mode}")

    # 1. Load Sequences
    seq_dict = read_fasta_to_dict(args.fasta)
    fasta_ids = list(seq_dict.keys())

    # 2. Prepare Results Container
    rows = []

    # 3. Mode: POOLED
    if args.mode == "pooled":
        if not args.pooled_cache or not args.supervised_csv:
            raise ValueError("Pooled mode requires --pooled-cache and --supervised-csv (for alignment).")

        # Load Cache
        X = _load_pooled_cache(args.pooled_cache)
        sup_ids = _load_supervised_ids(args.supervised_csv)

        # Align
        sup_idx = {pid: i for i, pid in enumerate(sup_ids)}
        keep_ids = [pid for pid in fasta_ids if pid in sup_idx]
        idxs = [sup_idx[pid] for pid in keep_ids]

        X_sel = X[idxs].float().to(device)

        # Model
        model = HeadOnlyModel(embed_dim=X_sel.shape[1], task=args.task, num_classes=args.num_classes).to(device)
        model.head.load_state_dict(torch.load(args.head, map_location=device))
        model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, len(X_sel), args.batch_size), desc="Predicting (Pooled)"):
                batch_x = X_sel[i:i + args.batch_size]
                out = model(batch_x)

                # Store results
                batch_ids = keep_ids[i:i + args.batch_size]
                if args.task == "classification":
                    probs = F.softmax(out, dim=1).cpu().numpy()
                    preds = probs.argmax(axis=1)
                    for k, pid in enumerate(batch_ids):
                        rows.append({"id": pid, "pred_class": int(preds[k]), "prob": probs[k].tolist()})
                else:
                    vals = out.cpu().numpy()
                    for k, pid in enumerate(batch_ids):
                        rows.append({"id": pid, "pred_value": float(vals[k])})

    # 4. Mode: TOKENS (Transformers)
    elif args.mode == "tokens":
        print(f"Loading HF Model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        base_model = AutoModel.from_pretrained(args.model_name)
        base_model.to(device).eval()

        # Head Model matches RepsHead structure (Backbone -> Pooler -> Head)
        # We manually stitch: Backbone(HF) + RepsHeadModel
        embed_dim = base_model.config.hidden_size
        head_model = RepsHeadModel(embed_dim, args.task, args.num_classes).to(device)
        head_model.head.load_state_dict(torch.load(args.head, map_location=device))
        head_model.eval()

        id_chunks = chunk_list(fasta_ids, args.esm_batch_size)

        for chunk in tqdm(id_chunks, desc="Predicting (Tokens)"):
            seqs = [seq_dict[pid][:args.max_len] for pid in chunk]

            # Tokenize
            inputs = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward
            with torch.no_grad():  # Add grad check if implementing Saliency
                out_base = base_model(**inputs)
                reps = out_base.last_hidden_state  # (B, L, D)
                mask = inputs["attention_mask"]

                # Head
                logits = head_model(reps, mask)

            # Store
            if args.task == "classification":
                probs = F.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)
                for k, pid in enumerate(chunk):
                    rows.append({"id": pid, "pred_class": int(preds[k]), "prob": probs[k].tolist()})
            else:
                vals = logits.cpu().numpy()
                for k, pid in enumerate(chunk):
                    rows.append({"id": pid, "pred_value": float(vals[k])})

    # Save
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--mode", default="pooled", choices=["pooled", "reps", "tokens"])
    parser.add_argument("--model-name", default="facebook/esm2_t33_650M_UR50D")

    # Cache args
    parser.add_argument("--pooled-cache")
    parser.add_argument("--supervised-csv")  # for alignment

    # Config
    parser.add_argument("--task", default="classification")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=1022)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--esm-batch-size", type=int, default=4)

    # Meta (optional shortcut)
    parser.add_argument("--meta")

    args = parser.parse_args()

    # Helper: Load from meta if provided
    if args.meta:
        meta = json.loads(Path(args.meta).read_text())
        cfg = meta.get("config", {})
        artifacts = meta.get("artifacts", {})
        cache_paths = meta.get("cache_paths", {})

        args.model_name = cfg.get("model_name", args.model_name)
        args.task = cfg.get("task", args.task)

        if not args.pooled_cache and "pooled" in cache_paths:
            args.pooled_cache = cache_paths["pooled"]
        if not args.supervised_csv and "supervised_csv" in artifacts:
            args.supervised_csv = artifacts["supervised_csv"]

    predict(args)


if __name__ == "__main__":
    main()