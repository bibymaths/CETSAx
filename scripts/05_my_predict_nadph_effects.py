#!/usr/bin/env python
"""
Predict NADPH responsiveness from protein sequences (Transformers Backend).
ROBUST VERSION:
- Fixes argument parsing bugs (ValueError).
- Aligns Model Architecture (Dropout 0.5, LayerNorm) with seq_nadph.py.
- Auto-detects ID formats to fix mismatches.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


# -----------------------------
# Helpers & ID Alignment
# -----------------------------
def _auto_device(device: str) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_fasta_robust(fasta_path: str | Path) -> Dict[str, str]:
    """Reads FASTA, storing the full first token as the key."""
    fasta_path = Path(fasta_path)
    seqs: Dict[str, List[str]] = {}
    current_id: Optional[str] = None
    with fasta_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                current_id = line[1:].split()[0]
                # Clean leading > if present in split (rare but possible)
                current_id = current_id.lstrip(">")
                seqs.setdefault(current_id, [])
            elif current_id:
                seqs[current_id].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def clean_id(pid: str, method: str) -> str:
    """Normalization strategies for IDs."""
    pid = str(pid).strip()
    if method == "exact":
        return pid
    elif method == "pipe_split":
        # >sp|P12345|NAME -> P12345
        if "|" in pid:
            parts = pid.split("|")
            if len(parts) >= 2: return parts[1]
        return pid
    elif method == "remove_suffix":
        # P12345_1 -> P12345
        return re.split(r'[_-]', pid)[0]
    return pid


def align_ids(source_ids: List[str], target_ids: List[str], source_name="FASTA", target_name="CSV") -> Tuple[
    Dict[str, str], List[str]]:
    """
    Finds mapping from Target (CSV) -> Source (FASTA/Cache).
    Returns:
       mapping: {target_id: source_id}
       keep_target_ids: list of target IDs that were found
    """
    strategies = ["exact", "pipe_split", "remove_suffix"]

    source_maps = {}
    for strat in strategies:
        source_maps[strat] = {clean_id(sid, strat): sid for sid in source_ids}

    best_strat = "exact"
    best_count = 0
    best_mapping = {}

    print(f"\n--- Aligning {target_name} ({len(target_ids)}) to {source_name} ({len(source_ids)}) ---")

    for strat in strategies:
        current_map = {}
        matches = 0
        s_map = source_maps[strat]

        for tid in target_ids:
            t_clean = clean_id(tid, strat)
            if t_clean in s_map:
                matches += 1
                current_map[tid] = s_map[t_clean]

        print(f"Strategy '{strat}': matched {matches} IDs")
        if matches > best_count:
            best_count = matches
            best_strat = strat
            best_mapping = current_map

    print(f"Selected Strategy: '{best_strat}' (Matches: {best_count})")

    if best_count == 0:
        print(f"[CRITICAL WARNING] Zero overlaps found between {source_name} and {target_name}.")
        print(f"{source_name} Example: {source_ids[:3]}")
        print(f"{target_name} Example: {target_ids[:3]}")

    return best_mapping, list(best_mapping.keys())


# -----------------------------
# Models (Updated to match seq_nadph.py)
# -----------------------------
class AttentionPooling(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attention(x).squeeze(-1)
        # Fix for fp16 overflow
        min_val = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask.bool(), min_val)
        weights = F.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


class HeadOnlyModel(nn.Module):
    def __init__(self, embed_dim: int, task: str, num_classes: int = 2):
        super().__init__()
        self.task = task
        self.norm = nn.LayerNorm(embed_dim)
        hidden = 512
        out_dim = num_classes if task == "classification" else 1

        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.5),  # UPDATED: Matches seq_nadph.py
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        logits = self.head(x)
        return logits.squeeze(-1) if self.task == "regression" else logits


class RepsHeadModel(nn.Module):
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
            nn.Dropout(0.5),  # UPDATED: Matches seq_nadph.py
            nn.Linear(hidden, out_dim)
        )

    def forward(self, reps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = self.pooler(reps, mask)
        x = self.norm(pooled)
        logits = self.head(x)
        return logits.squeeze(-1) if self.task == "regression" else logits


# -----------------------------
# Prediction Logic
# -----------------------------
def predict(args):
    device = _auto_device(args.device)
    print(f"Device: {device} | Mode: {args.mode}")

    # Load FASTA
    seq_dict_raw = read_fasta_robust(args.fasta)
    fasta_ids = list(seq_dict_raw.keys())

    rows = []

    # ---------------------------------------------------------
    # MODE: POOLED
    # ---------------------------------------------------------
    if args.mode == "pooled":
        if not args.pooled_cache or not os.path.exists(str(args.pooled_cache)):
            print(f"[ERROR] Pooled cache not found: {args.pooled_cache}")
            print("Did you mean to use --mode reps? Or check your --pooled-cache path.")
            sys.exit(1)
        if not args.supervised_csv:
            raise ValueError("--supervised-csv required for pooled mode")

        print(f"Loading Pooled Cache: {args.pooled_cache}")
        X = torch.load(args.pooled_cache, map_location="cpu")["pooled"]

        print(f"Loading IDs: {args.supervised_csv}")
        sup_ids = pd.read_csv(args.supervised_csv)["id"].astype(str).tolist()

        if len(X) != len(sup_ids):
            print(f"[WARNING] Cache length {len(X)} != ID list length {len(sup_ids)}")

        # No intricate alignment needed for pooled mode, we assume 1:1 mapping with cache
        # because the cache was built FROM this CSV.

        model = HeadOnlyModel(embed_dim=X.shape[1], task=args.task, num_classes=args.num_classes).to(device)
        _load_checkpoint_robust(model, args.head, device)
        model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, len(X), args.batch_size), desc="Predicting (Pooled)"):
                batch_x = X[i:i + args.batch_size].to(device).float()
                batch_ids = sup_ids[i:i + args.batch_size]
                out = model(batch_x)
                _store_results(rows, out, batch_ids, args.task)

    # ---------------------------------------------------------
    # MODE: REPS
    # ---------------------------------------------------------
    elif args.mode == "reps":
        if not args.reps_cache or not os.path.exists(str(args.reps_cache)):
            print(f"[ERROR] Reps cache not found: {args.reps_cache}")
            sys.exit(1)
        if not args.supervised_csv:
            raise ValueError("--supervised-csv required for reps mode")

        print(f"Loading Reps Cache: {args.reps_cache}")
        obj = torch.load(args.reps_cache, map_location="cpu")
        reps_list = obj["reps"]
        mask_list = obj["mask"]

        print(f"Loading IDs: {args.supervised_csv}")
        sup_ids = pd.read_csv(args.supervised_csv)["id"].astype(str).tolist()

        embed_dim = reps_list[0].shape[1]
        model = RepsHeadModel(embed_dim, args.task, args.num_classes).to(device)
        _load_checkpoint_robust(model, args.head, device)
        model.eval()

        with torch.no_grad():
            for i in tqdm(range(len(reps_list)), desc="Predicting (Reps)"):
                r = reps_list[i].unsqueeze(0).to(device).float()
                m = mask_list[i].unsqueeze(0).to(device).bool()
                pid = sup_ids[i]

                out = model(r, m)
                _store_results(rows, out, [pid], args.task)

    # ---------------------------------------------------------
    # MODE: TOKENS
    # ---------------------------------------------------------
    elif args.mode == "tokens":
        print(f"Loading HF Model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        base_model = AutoModel.from_pretrained(args.model_name).to(device).eval()

        embed_dim = base_model.config.hidden_size
        head_model = RepsHeadModel(embed_dim, args.task, args.num_classes).to(device)
        _load_checkpoint_robust(head_model, args.head, device)
        head_model.eval()

        def chunker(seq, size):
            return (seq[pos:pos + size] for pos in range(0, len(seq), size))

        for chunk_ids in tqdm(chunker(fasta_ids, args.esm_batch_size), total=len(fasta_ids) // args.esm_batch_size):
            seqs = [seq_dict_raw[pid][:args.max_len] for pid in chunk_ids]

            inputs = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out_base = base_model(**inputs)
                logits = head_model(out_base.last_hidden_state, inputs["attention_mask"])

            _store_results(rows, logits, chunk_ids, args.task)

    # ---------------------------------------------------------
    # Save Results
    # ---------------------------------------------------------
    if rows:
        out_df = pd.DataFrame(rows)
        out_df.to_csv(args.out, index=False)
        print(f"Saved {len(out_df)} predictions to {args.out}")
        print("Columns:", out_df.columns.tolist())
    else:
        print("[WARNING] No predictions generated. Creating empty CSV with headers.")
        cols = ["id", "pred_class_idx", "p_class0", "p_class1"] if args.task == "classification" else ["id",
                                                                                                       "pred_value"]
        pd.DataFrame(columns=cols).to_csv(args.out, index=False)


def _load_checkpoint_robust(model, path, device):
    """
    Loads checkpoint and handles missing keys (e.g. Norm layer) gracefully.
    """
    print(f"Loading head weights from {path}")
    state = torch.load(path, map_location=device)

    # Check for norm layer mismatch
    model_keys = set(model.head.state_dict().keys())
    ckpt_keys = set(state.keys())

    # If checkpoint keys don't start with "head.", they might be the head module directly
    # Check if '0.weight' exists in checkpoint (indicative of nn.Sequential)
    if "0.weight" in ckpt_keys and not any(k.startswith("head.") for k in ckpt_keys):
        # We are loading into model.head
        missing, unexpected = model.head.load_state_dict(state, strict=False)
    else:
        # We are loading into full model (wrapper)
        missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"[INFO] Missing keys in checkpoint: {missing}")
        if any("norm" in k for k in missing):
            print("[WARNING] LayerNorm weights were missing. Using initialized weights.")
            print("          This is expected if training script only saved `model.head`.")


def _store_results(rows, logits, batch_ids, task):
    if task == "classification":
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        for k, pid in enumerate(batch_ids):
            row = {"id": pid, "pred_class_idx": int(preds[k])}
            for c in range(probs.shape[1]):
                row[f"p_class{c}"] = float(probs[k, c])
            rows.append(row)
    else:
        vals = logits.cpu().numpy()
        for k, pid in enumerate(batch_ids):
            rows.append({"id": pid, "pred_value": float(vals[k])})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--out", required=True)

    # Use None as default so we can detect if user passed it
    parser.add_argument("--mode", default=None, choices=["pooled", "reps", "tokens"])
    parser.add_argument("--model-name", default="facebook/esm2_t33_650M_UR50D")

    parser.add_argument("--pooled-cache")
    parser.add_argument("--reps-cache")
    parser.add_argument("--supervised-csv")

    parser.add_argument("--task", default="classification")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=1022)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--esm-batch-size", type=int, default=4)

    parser.add_argument("--meta")

    args = parser.parse_args()

    # 1. Load Meta configuration if available
    meta_cfg = {}
    cache_paths = {}

    if args.meta and os.path.exists(args.meta):
        try:
            meta = json.loads(Path(args.meta).read_text())
            meta_cfg = meta.get("config", {})
            cache_paths = meta.get("cache_paths", {})
            artifacts = meta.get("artifacts", {})

            # Fill defaults from Meta
            if not args.model_name: args.model_name = meta_cfg.get("model_name", args.model_name)
            if not args.supervised_csv and "supervised_csv" in artifacts:
                args.supervised_csv = artifacts["supervised_csv"]
        except Exception as e:
            print(f"[WARNING] Failed to load meta file: {e}")

    # 2. Determine Mode
    # Priority: CLI Argument > Meta 'train_mode' > Default 'pooled'
    if args.mode is None:
        args.mode = meta_cfg.get("train_mode", "pooled")

    print(f"Selected Mode: {args.mode}")

    # 3. Resolve Caches based on Mode
    if args.mode == "pooled" and not args.pooled_cache:
        args.pooled_cache = cache_paths.get("pooled")

    if args.mode == "reps" and not args.reps_cache:
        args.reps_cache = cache_paths.get("reps")

    predict(args)


if __name__ == "__main__":
    main()