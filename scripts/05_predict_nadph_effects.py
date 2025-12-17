#!/usr/bin/env python
"""
Script to predict NADPH responsiveness from protein sequences using a trained head.

Supports inference modes:
- pooled: pooled embedding -> head (fast). Uses cache if provided.
- reps: residue reps (+mask) -> head with attention pooling (supports saliency/IG if reps cached).
- tokens: tokens -> ESM -> pooling -> head (slowest, but simplest).

For 1x15GB GPU + ESM2-650M, ESM forward batch size is auto-clamped to 1-2.
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
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from cetsax.deeplearn.seq_nadph import (
    read_fasta_to_dict,
    load_esm_model_and_alphabet,
    AttentionPooling,
    HeadOnlyModel,
    compute_residue_saliency,
    compute_residue_integrated_gradients,
)

# -----------------------------
# Helpers
# -----------------------------

def chunk_list(xs: List, size: int) -> List[List]:
    return [xs[i:i + size] for i in range(0, len(xs), size)]

def _auto_device(device: str) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _safe_esm_batch_size(device: torch.device, requested: int) -> int:
    # ESM2-650M on 15GB GPU typically needs bs=1-2 for long sequences.
    if device.type == "cuda":
        return max(1, min(2, requested))
    return max(1, requested)

def _infer_int_to_class(num_classes: int) -> Dict[int, str]:
    # default mapping; customize if you have >2 classes
    if num_classes == 2:
        return {0: "weak", 1: "strong"}
    return {i: f"class_{i}" for i in range(num_classes)}

# -----------------------------
# Models for prediction modes
# -----------------------------

class RepsHeadModel(nn.Module):
    """
    Uses cached residue reps (B,L,D) + mask (B,L) and applies
    AttentionPooling + LayerNorm + MLP head (same idea as training).
    This does NOT require ESM forward at prediction time.
    """
    def __init__(self, embed_dim: int, task: str, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.task = task
        self.pooler = AttentionPooling(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        hidden = 256
        out_dim = num_classes if task == "classification" else 1
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, reps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = self.pooler(reps, mask)
        pooled = self.norm(pooled)
        logits = self.head(pooled)
        if self.task == "regression":
            logits = logits.squeeze(-1)
        return logits


# -----------------------------
# Cache readers (produced by training module)
# -----------------------------

def _load_pooled_cache(emb_pt: str | Path) -> Tuple[List[str], torch.Tensor]:
    """
    Expected format from training:
      {"ids": [..], "embs": (N,D) tensor, ...}
    If your cache doesn't store ids, you must align externally.
    """
    obj = torch.load(emb_pt, map_location="cpu")
    if "ids" not in obj:
        raise ValueError(f"Pooled cache {emb_pt} has no 'ids' list. Save ids during caching.")
    ids = list(obj["ids"])
    x = obj["embs"]  # (N,D) fp16/fp32
    return ids, x

def _load_reps_cache(reps_pt: str | Path) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
    """
    Expected format:
      {"ids": [..], "reps": list[(L,D)], "mask": list[(L,)]}
    (Storing lists avoids padding huge (N,L,D) tensors.)
    """
    obj = torch.load(reps_pt, map_location="cpu")
    for k in ("ids", "reps", "mask"):
        if k not in obj:
            raise ValueError(f"Reps cache {reps_pt} missing key '{k}'.")
    return list(obj["ids"]), obj["reps"], obj["mask"]

# -----------------------------
# Main prediction function
# -----------------------------

def predict_nadph_from_fasta(
    fasta_path: str | Path,
    head_checkpoint: str | Path,
    mode: str = "pooled",  # pooled | reps | tokens
    model_name: str = "esm2_t33_650M_UR50D",
    task: str = "classification",
    num_classes: int = 2,
    max_len: int = 1022,
    device: str = "auto",

    # batching
    batch_size: int = 256,      # for head-only (pooled/reps) inference
    esm_batch_size: int = 8,    # for ESM forward in tokens-mode (auto-clamped on cuda)

    # optional caches (from training meta.json)
    pooled_cache_pt: Optional[str | Path] = None,
    reps_cache_pt: Optional[str | Path] = None,

    # interpretability
    compute_saliency: bool = False,
    compute_ig: bool = False,
    target_class: Optional[int] = None,
    ig_steps: int = 50,
) -> pd.DataFrame:
    fasta_path = Path(fasta_path)
    head_checkpoint = Path(head_checkpoint)
    device_t = _auto_device(device)

    if mode not in ("pooled", "reps", "tokens"):
        raise ValueError("mode must be one of: pooled, reps, tokens")

    if (compute_saliency or compute_ig) and mode == "pooled":
        raise ValueError("Saliency/IG requires residue-level reps. Use mode=reps or mode=tokens.")

    # 1) Load sequences
    seq_dict: Dict[str, str] = read_fasta_to_dict(fasta_path)
    ids = list(seq_dict.keys())
    if not ids:
        raise ValueError(f"No sequences found in FASTA: {fasta_path}")

    print(f"Using device: {device_t}")
    print(f"Loaded {len(ids)} sequences from {fasta_path.name}")
    print(f"Mode: {mode} | task={task} | num_classes={num_classes}")

    int_to_class = _infer_int_to_class(num_classes)

    # -------------------------
    # MODE: pooled (fast)
    # -------------------------
    if mode == "pooled":
        # Option A: use pooled cache (recommended)
        if pooled_cache_pt is None:
            raise ValueError("mode=pooled requires --pooled-cache (embeddings .pt).")

        cache_ids, X = _load_pooled_cache(pooled_cache_pt)
        # Align cache to FASTA ids (only predict on intersection)
        cache_map = {pid: i for i, pid in enumerate(cache_ids)}
        keep = [pid for pid in ids if pid in cache_map]
        if not keep:
            raise ValueError("No FASTA ids matched pooled cache ids.")
        idxs = [cache_map[pid] for pid in keep]
        X_sel = X[idxs].float()  # train in float32 for stability

        # Build head-only model and load weights
        embed_dim = X_sel.shape[1]
        model = HeadOnlyModel(embed_dim=embed_dim, task=task, num_classes=num_classes).to(device_t)
        state = torch.load(head_checkpoint, map_location=device_t)
        model.load_state_dict(state, strict=True)
        model.eval()

        rows = []
        with torch.no_grad():
            for j0 in tqdm(range(0, len(keep), batch_size), desc="Predict (pooled)"):
                j1 = min(len(keep), j0 + batch_size)
                xb = X_sel[j0:j1].to(device_t)
                out = model(xb)

                if task == "classification":
                    probs = F.softmax(out, dim=1).detach().cpu().numpy()
                    preds = probs.argmax(axis=1)
                    for k, pid in enumerate(keep[j0:j1]):
                        row = {"id": pid, "pred_class_idx": int(preds[k]), "pred_class": int_to_class.get(int(preds[k]), f"class_{int(preds[k])}")}
                        for c in range(probs.shape[1]):
                            row[f"p_class{c}"] = float(probs[k, c])
                        rows.append(row)
                else:
                    vals = out.detach().cpu().numpy()
                    for k, pid in enumerate(keep[j0:j1]):
                        rows.append({"id": pid, "pred_value": float(vals[k])})

        return pd.DataFrame(rows)

    # -------------------------
    # MODE: reps (supports saliency/IG if reps cache exists)
    # -------------------------
    if mode == "reps":
        if reps_cache_pt is None:
            raise ValueError("mode=reps requires --reps-cache (reps .pt).")

        cache_ids, reps_list, mask_list = _load_reps_cache(reps_cache_pt)
        cache_map = {pid: i for i, pid in enumerate(cache_ids)}
        keep = [pid for pid in ids if pid in cache_map]
        if not keep:
            raise ValueError("No FASTA ids matched reps cache ids.")

        # infer embed_dim
        ex = reps_list[cache_map[keep[0]]]
        embed_dim = int(ex.shape[-1])

        model = RepsHeadModel(embed_dim=embed_dim, task=task, num_classes=num_classes).to(device_t)
        state = torch.load(head_checkpoint, map_location=device_t)
        # checkpoint in reps-mode should be model.head state OR full head-only state; here we expect full head.
        # If you saved only model.head.state_dict(), then load into model.head instead of model.
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            model.head.load_state_dict(state, strict=True)

        model.eval()

        rows = []
        # For saliency/IG with cached reps: we need gradients through reps -> head.
        grad_enabled = (compute_saliency or compute_ig)
        ctx = torch.enable_grad if grad_enabled else torch.no_grad

        with ctx():
            # batch pad reps/mask for efficiency
            for j0 in tqdm(range(0, len(keep), batch_size), desc="Predict (reps)"):
                batch_ids = keep[j0:j0 + batch_size]
                reps_batch = [reps_list[cache_map[pid]] for pid in batch_ids]
                mask_batch = [mask_list[cache_map[pid]] for pid in batch_ids]

                # pad to max L in batch
                Lmax = max(int(r.shape[0]) for r in reps_batch)
                D = embed_dim

                reps_pad = torch.zeros((len(batch_ids), Lmax, D), dtype=reps_batch[0].dtype)
                mask_pad = torch.zeros((len(batch_ids), Lmax), dtype=torch.bool)

                for i, (r, m) in enumerate(zip(reps_batch, mask_batch)):
                    L = int(r.shape[0])
                    reps_pad[i, :L] = r
                    mask_pad[i, :L] = m.to(torch.bool) if isinstance(m, torch.Tensor) else torch.tensor(m, dtype=torch.bool)

                reps_pad = reps_pad.to(device_t).float()
                mask_pad = mask_pad.to(device_t)

                if grad_enabled:
                    reps_pad.requires_grad_(True)

                out = model(reps_pad, mask_pad)

                # base preds
                if task == "classification":
                    probs = F.softmax(out, dim=1)
                    preds = probs.argmax(dim=1)
                    probs_np = probs.detach().cpu().numpy()
                    preds_np = preds.detach().cpu().numpy()
                else:
                    preds_np = out.detach().cpu().numpy()

                # saliency/IG on cached reps:
                # - "saliency": grad norm of reps wrt selected output (same idea as your existing function)
                # - "IG": integrated gradients in reps-space (heavy; only do if you really need it)
                sal_scores = None
                ig_scores = None

                if compute_saliency:
                    model.zero_grad(set_to_none=True)
                    if task == "classification":
                        if target_class is None:
                            sel = out.gather(1, preds.unsqueeze(1)).squeeze(1)
                        else:
                            sel = out[:, int(target_class)]
                        loss = sel.sum()
                    else:
                        loss = out.sum()
                    loss.backward()
                    sal = reps_pad.grad.norm(dim=-1)  # (B,L)
                    sal = sal * mask_pad.float()
                    sal_scores = sal.detach().cpu()

                if compute_ig:
                    # Integrated gradients in reps-space (expensive).
                    # For practical use: keep ig_steps small (e.g., 20).
                    base = torch.zeros_like(reps_pad)
                    total_grad = torch.zeros_like(reps_pad)
                    for a in torch.linspace(0.0, 1.0, ig_steps, device=device_t):
                        model.zero_grad(set_to_none=True)
                        x = base + a * (reps_pad.detach() - base)
                        x.requires_grad_(True)
                        o = model(x, mask_pad)
                        if task == "classification":
                            if target_class is None:
                                sel = o.gather(1, preds.unsqueeze(1)).squeeze(1)
                            else:
                                sel = o[:, int(target_class)]
                            loss = sel.sum()
                        else:
                            loss = o.sum()
                        loss.backward()
                        total_grad += x.grad
                    avg_grad = total_grad / float(ig_steps)
                    ig = (reps_pad.detach() - base) * avg_grad
                    ig_scores = ig.norm(dim=-1) * mask_pad.float()
                    ig_scores = ig_scores.detach().cpu()

                # assemble rows
                for i, pid in enumerate(batch_ids):
                    row = {"id": pid}
                    if task == "classification":
                        pc = int(preds_np[i])
                        row["pred_class_idx"] = pc
                        row["pred_class"] = int_to_class.get(pc, f"class_{pc}")
                        for c in range(probs_np.shape[1]):
                            row[f"p_class{c}"] = float(probs_np[i, c])
                    else:
                        row["pred_value"] = float(preds_np[i])

                    # slice to true L using mask
                    if compute_saliency and sal_scores is not None:
                        L = int(mask_pad[i].sum().item())
                        s = sal_scores[i, :L].numpy()
                        row["saliency"] = ";".join(f"{x:.6f}" for x in s)

                    if compute_ig and ig_scores is not None:
                        L = int(mask_pad[i].sum().item())
                        g = ig_scores[i, :L].numpy()
                        row["ig"] = ";".join(f"{x:.6f}" for x in g)

                    rows.append(row)

        return pd.DataFrame(rows)

    # -------------------------
    # MODE: tokens (slow; runs ESM)
    # -------------------------
    # This mode doesn't need caches but does need ESM.
    device_t = _auto_device(device)
    esm_batch_size_eff = _safe_esm_batch_size(device_t, esm_batch_size)

    print(f"Loading ESM model: {model_name}")
    esm_model, alphabet = load_esm_model_and_alphabet(model_name)
    esm_model.to(device_t).eval()
    batch_converter = alphabet.get_batch_converter()

    embed_dim = esm_model.embed_dim
    # same RepsHeadModel but reps computed from ESM on the fly
    model = RepsHeadModel(embed_dim=embed_dim, task=task, num_classes=num_classes).to(device_t)
    state = torch.load(head_checkpoint, map_location=device_t)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.head.load_state_dict(state, strict=True)
    model.eval()

    # inference
    rows = []
    grad_enabled = (compute_saliency or compute_ig)
    ctx = torch.enable_grad if grad_enabled else torch.no_grad

    # For tokens-mode we chunk sequences then run ESM forward
    id_chunks = chunk_list(ids, esm_batch_size_eff)
    print(f"Starting inference: {len(id_chunks)} batches (esm_bs={esm_batch_size_eff})")

    with ctx():
        for batch_idx, id_chunk in enumerate(tqdm(id_chunks, desc="Predict (tokens)")):
            seq_chunk = []
            for pid in id_chunk:
                s = seq_dict[pid]
                if len(s) > max_len:
                    s = s[:max_len]
                seq_chunk.append((pid, s))

            _, _, toks = batch_converter(seq_chunk)
            toks = toks.to(device_t)
            mask = (toks != 1)

            # ESM forward (residue reps)
            out = esm_model(toks, repr_layers=[33], return_contacts=False)
            reps = out["representations"][33]  # (B,L,D)

            if grad_enabled:
                reps.retain_grad()

            logits = model(reps, mask)

            # predictions
            if task == "classification":
                probs = F.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                probs_np = probs.detach().cpu().numpy()
                preds_np = preds.detach().cpu().numpy()
            else:
                preds_np = logits.detach().cpu().numpy()

            # saliency/IG in tokens-mode: reuse your module functions if you prefer,
            # but those functions expect NADPHSeqModel(tokens)->(logits,reps,mask).
            # Here we compute directly on reps.
            sal_scores = None
            ig_scores = None

            if compute_saliency:
                model.zero_grad(set_to_none=True)
                if task == "classification":
                    if target_class is None:
                        sel = logits.gather(1, preds.unsqueeze(1)).squeeze(1)
                    else:
                        sel = logits[:, int(target_class)]
                    loss = sel.sum()
                else:
                    loss = logits.sum()
                loss.backward()
                sal = reps.grad.norm(dim=-1) * mask.float()
                sal_scores = sal.detach().cpu()

            if compute_ig:
                base = torch.zeros_like(reps)
                total_grad = torch.zeros_like(reps)
                for a in torch.linspace(0.0, 1.0, ig_steps, device=device_t):
                    model.zero_grad(set_to_none=True)
                    x = base + a * (reps.detach() - base)
                    x.requires_grad_(True)
                    o = model(x, mask)
                    if task == "classification":
                        if target_class is None:
                            sel = o.gather(1, preds.unsqueeze(1)).squeeze(1)
                        else:
                            sel = o[:, int(target_class)]
                        loss = sel.sum()
                    else:
                        loss = o.sum()
                    loss.backward()
                    total_grad += x.grad
                avg_grad = total_grad / float(ig_steps)
                ig = (reps.detach() - base) * avg_grad
                ig_scores = (ig.norm(dim=-1) * mask.float()).detach().cpu()

            # assemble rows
            for i, pid in enumerate(id_chunk):
                row = {"id": pid}
                if task == "classification":
                    pc = int(preds_np[i])
                    row["pred_class_idx"] = pc
                    row["pred_class"] = int_to_class.get(pc, f"class_{pc}")
                    for c in range(probs_np.shape[1]):
                        row[f"p_class{c}"] = float(probs_np[i, c])
                else:
                    row["pred_value"] = float(preds_np[i])

                # exact seq len from seq_chunk (string)
                seq_len = len(seq_chunk[i][1])
                # slice residues (skip <cls> and <eos>: ESM tokens contain those; reps align with tokens)
                # reps indexing typically matches token positions; safest slice: 1..seq_len
                if compute_saliency and sal_scores is not None:
                    s = sal_scores[i, 1: seq_len + 1].numpy()
                    row["saliency"] = ";".join(f"{x:.6f}" for x in s)
                if compute_ig and ig_scores is not None:
                    g = ig_scores[i, 1: seq_len + 1].numpy()
                    row["ig"] = ";".join(f"{x:.6f}" for x in g)

                rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Predict NADPH responsiveness from protein sequences using trained head.")

    parser.add_argument("--fasta", default="../results/protein_sequences.fasta",
                        help="FASTA file with protein sequences.")
    parser.add_argument("--head", dest="head_checkpoint", default="../results/nadph_seq_head.pt",
                        help="Trained head checkpoint (state_dict).")

    parser.add_argument("--mode", choices=["pooled", "reps", "tokens"], default="pooled",
                        help="Inference mode: pooled (fast), reps (cached reps), tokens (run ESM).")

    parser.add_argument("--meta", default=None,
                        help="Optional training meta JSON produced by training script (to auto-fill cache paths/model/task).")

    parser.add_argument("--pooled-cache", default=None,
                        help="Path to pooled embedding cache .pt (required for mode=pooled unless meta provides it).")
    parser.add_argument("--reps-cache", default=None,
                        help="Path to reps cache .pt (required for mode=reps unless meta provides it).")

    parser.add_argument("--model-name", default="esm2_t33_650M_UR50D",
                        help="ESM model name (used in tokens-mode).")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification",
                        help="Prediction task type.")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of classes (classification).")
    parser.add_argument("--max-len", type=int, default=1022,
                        help="Truncate sequences longer than this.")
    parser.add_argument("--device", default="auto",
                        help="cuda/cpu/auto")

    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for head-only inference (pooled/reps).")
    parser.add_argument("--esm-batch-size", type=int, default=8,
                        help="Requested batch size for ESM forward (tokens-mode). Auto-clamped on cuda to 1-2.")

    parser.add_argument("--saliency", action="store_true", help="Compute residue saliency (reps/tokens only).")
    parser.add_argument("--ig", action="store_true", help="Compute residue integrated gradients (reps/tokens only).")
    parser.add_argument("--target-class", type=int, default=None,
                        help="Target class for saliency/IG; if omitted uses predicted class.")
    parser.add_argument("--ig-steps", type=int, default=50, help="IG steps (keep small; expensive).")

    parser.add_argument("--out", default="../results/predictions_nadph_seq.csv",
                        help="Output CSV path.")

    args = parser.parse_args()

    # If meta is provided, auto-fill fields when missing
    if args.meta is not None:
        meta = json.loads(Path(args.meta).read_text())
        cfg = meta.get("config", {})
        cache_paths = meta.get("cache_paths", {})

        # only fill if user didn't override
        if args.task == "classification" and cfg.get("task") in ("classification", "regression"):
            args.task = cfg["task"]
        if args.model_name == "esm2_t33_650M_UR50D" and cfg.get("model_name"):
            args.model_name = cfg["model_name"]
        if args.num_classes == 2 and cfg.get("num_classes") is not None:
            args.num_classes = int(cfg["num_classes"])

        if args.pooled_cache is None and "pooled_cache_pt" in cache_paths:
            args.pooled_cache = cache_paths["pooled_cache_pt"]
        if args.reps_cache is None and "reps_cache_pt" in cache_paths:
            args.reps_cache = cache_paths["reps_cache_pt"]

    df_pred = predict_nadph_from_fasta(
        fasta_path=args.fasta,
        head_checkpoint=args.head_checkpoint,
        mode=args.mode,
        model_name=args.model_name,
        task=args.task,
        num_classes=args.num_classes,
        max_len=args.max_len,
        device=args.device,
        batch_size=args.batch_size,
        esm_batch_size=args.esm_batch_size,
        pooled_cache_pt=args.pooled_cache,
        reps_cache_pt=args.reps_cache,
        compute_saliency=args.saliency,
        compute_ig=args.ig,
        target_class=args.target_class,
        ig_steps=args.ig_steps,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path} (n={len(df_pred)})")


if __name__ == "__main__":
    main()