#!/usr/bin/env python
"""
Predict NADPH responsiveness from protein sequences using a trained head.

Modes:
- pooled: uses cached pooled embeddings (fastest)
- reps:   uses cached residue reps + mask (fast, supports saliency/IG without ESM)
- tokens: runs ESM forward (slow, supports saliency/IG)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from cetsax.deeplearn.esm_seq_nadph import (
    read_fasta_to_dict,
    load_esm_model_and_alphabet,
    AttentionPooling,
    NADPHSeqModel,
    compute_residue_saliency_from_reps,
    compute_residue_integrated_gradients_from_reps,
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
    # ESM2-650M on ~15GB GPU often needs bs=1-2 for long sequences.
    if device.type == "cuda":
        return max(1, min(2, requested))
    return max(1, requested)


def _infer_int_to_class(num_classes: int) -> Dict[int, str]:
    if num_classes == 2:
        return {0: "weak", 1: "strong"}
    return {i: f"class_{i}" for i in range(num_classes)}


def _get_head_module(model: nn.Module) -> nn.Module:
    """
    Mirror the helper you already wrote in the module.
    """
    base = model.module if hasattr(model, "module") else model
    if hasattr(base, "head") and isinstance(getattr(base, "head"), nn.Module):
        return base.head
    if hasattr(base, "net") and isinstance(getattr(base, "net"), nn.Module):
        return base.net
    raise AttributeError(f"Could not find head on model type={type(base)}")


class HeadOnlyModel(nn.Module):
    """
    Minimal head-only wrapper for pooled embeddings.
    Matches NADPHSeqModel.head output behavior.
    """
    def __init__(self, embed_dim: int, task: str, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.task = task
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        logits = self.head(x)
        if self.task == "regression":
            logits = logits.squeeze(-1)
        return logits


class RepsHeadModel(nn.Module):
    """
    Uses cached residue reps (B,L,D) + mask (B,L) and applies
    AttentionPooling + LayerNorm + MLP head.
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
# Cache readers
# -----------------------------

def _load_pooled_cache(pooled_pt: str | Path) -> Tuple[List[str], torch.Tensor]:
    obj = torch.load(pooled_pt, map_location="cpu", weights_only=False)
    if "ids" not in obj or "pooled" not in obj:
        raise ValueError(f"{pooled_pt} must contain keys 'ids' and 'pooled'. Keys={list(obj.keys())}")
    return [str(x) for x in obj["ids"]], obj["pooled"]


def _load_reps_cache(reps_pt: str | Path) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
    obj = torch.load(reps_pt, map_location="cpu", weights_only=False)
    for k in ("ids", "reps", "mask"):
        if k not in obj:
            raise ValueError(f"{reps_pt} missing key '{k}'. Keys={list(obj.keys())}")
    return [str(x) for x in obj["ids"]], obj["reps"], obj["mask"]


# def _load_pooled_cache(pooled_pt: str | Path) -> torch.Tensor:
#     """
#     Your module writes: {"pooled": (N,D), "label_cls":..., "label_reg"?:...}
#     """
#     obj = torch.load(pooled_pt, map_location="cpu")
#     if "pooled" not in obj:
#         raise ValueError(f"Pooled cache {pooled_pt} missing key 'pooled'. Keys={list(obj.keys())}")
#     return obj["pooled"]  # (N,D) fp16/fp32
#
#
# def _load_reps_cache(reps_pt: str | Path) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
#     """
#     Your module writes: {"reps": list[(L,D)], "mask": list[(L,)], ...}
#     """
#     obj = torch.load(reps_pt, map_location="cpu")
#     for k in ("reps", "mask"):
#         if k not in obj:
#             raise ValueError(f"Reps cache {reps_pt} missing key '{k}'. Keys={list(obj.keys())}")
#     return obj["reps"], obj["mask"]


# def _load_supervised_ids(supervised_csv: str | Path) -> List[str]:
#     df = pd.read_csv(supervised_csv)
#     if "id" not in df.columns:
#         raise ValueError(f"{supervised_csv} missing column 'id'")
#     return df["id"].astype(str).tolist()


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
    repr_layer: int = 33,
    device: str = "auto",

    # batching
    batch_size: int = 256,      # for head-only (pooled/reps)
    esm_batch_size: int = 8,    # tokens-mode (auto-clamped on cuda)

    # caches + alignment
    pooled_cache_pt: Optional[str | Path] = None,
    reps_cache_pt: Optional[str | Path] = None,
    supervised_csv_for_alignment: Optional[str | Path] = None,

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
    fasta_ids = list(seq_dict.keys())
    if not fasta_ids:
        raise ValueError(f"No sequences found in FASTA: {fasta_path}")

    print(f"Using device: {device_t}")
    print(f"Loaded {len(fasta_ids)} sequences from {fasta_path.name}")
    print(f"Mode: {mode} | task={task} | num_classes={num_classes}")

    int_to_class = _infer_int_to_class(num_classes)

    # -------------------------
    # MODE: pooled (fast)
    # -------------------------
    # if mode == "pooled":
    #     if pooled_cache_pt is None:
    #         raise ValueError("mode=pooled requires pooled_cache_pt (use --pooled-cache or provide meta.json).")
    #     if supervised_csv_for_alignment is None:
    #         raise ValueError(
    #             "mode=pooled requires supervised_csv_for_alignment because your pooled cache does not store ids. "
    #             "Provide --meta so we can read artifacts.supervised_csv, or pass --supervised-csv."
    #         )
    #
    #     # load pooled embeddings (N,D) and supervised ids (N,)
    #     X = _load_pooled_cache(pooled_cache_pt)
    #     sup_ids = _load_supervised_ids(supervised_csv_for_alignment)
    #
    #     if len(sup_ids) != int(X.shape[0]):
    #         raise ValueError(
    #             f"Alignment mismatch: supervised has n={len(sup_ids)} but pooled cache has N={int(X.shape[0])}. "
    #             "They must match because we align by row order."
    #         )
    #
    #     # map from id -> row index (supervised order)
    #     sup_index = {pid: i for i, pid in enumerate(sup_ids)}
    #
    #     keep = [pid for pid in fasta_ids if pid in sup_index]
    #     if not keep:
    #         raise ValueError("No FASTA ids matched supervised ids (used to align pooled cache).")
    #
    #     idxs = [sup_index[pid] for pid in keep]
    #     X_sel = X[idxs].float()  # use fp32 for inference stability
    #
    #     # head-only model
    #     embed_dim = int(X_sel.shape[1])
    #     model = HeadOnlyModel(embed_dim=embed_dim, task=task, num_classes=num_classes).to(device_t)
    #
    #     state = torch.load(head_checkpoint, map_location=device_t)
    #     model.head.load_state_dict(state, strict=True)  # you saved head.state_dict()
    #     model.eval()
    #
    #     rows = []
    #     with torch.no_grad():
    #         for j0 in tqdm(range(0, len(keep), batch_size), desc="Predict (pooled)"):
    #             j1 = min(len(keep), j0 + batch_size)
    #             xb = X_sel[j0:j1].to(device_t)
    #             out = model(xb)
    #
    #             if task == "classification":
    #                 probs = F.softmax(out, dim=1).cpu().numpy()
    #                 preds = probs.argmax(axis=1)
    #                 for k, pid in enumerate(keep[j0:j1]):
    #                     pc = int(preds[k])
    #                     row = {"id": pid, "pred_class_idx": pc, "pred_class": int_to_class.get(pc, f"class_{pc}")}
    #                     for c in range(probs.shape[1]):
    #                         row[f"p_class{c}"] = float(probs[k, c])
    #                     rows.append(row)
    #             else:
    #                 vals = out.cpu().numpy()
    #                 for k, pid in enumerate(keep[j0:j1]):
    #                     rows.append({"id": pid, "pred_value": float(vals[k])})
    #
    #     return pd.DataFrame(rows)

    if mode == "pooled":
        if pooled_cache_pt is None:
            raise ValueError("mode=pooled requires pooled_cache_pt (use --pooled-cache or --meta).")

        ids, X = _load_pooled_cache(pooled_cache_pt)
        id_to_idx = {pid: i for i, pid in enumerate(ids)}

        keep = [pid for pid in fasta_ids if pid in id_to_idx]
        if not keep:
            raise ValueError("No FASTA ids matched pooled cache ids.")

        idxs = [id_to_idx[pid] for pid in keep]
        X_sel = X[idxs].float()  # fp32 for stable inference

        embed_dim = int(X_sel.shape[1])
        model = NADPHSeqModel(esm_model=None, embed_dim=embed_dim, task=task, num_classes=num_classes).to(device_t)

        state = torch.load(head_checkpoint, map_location=device_t, weights_only = False)
        model.head.load_state_dict(state, strict=True)
        model.eval()

        rows = []
        with torch.no_grad():
            for j0 in tqdm(range(0, len(keep), batch_size), desc="Predict (pooled)"):
                j1 = min(len(keep), j0 + batch_size)
                xb = X_sel[j0:j1].to(device_t)
                out = model(pooled=xb)

                if task == "classification":
                    probs = F.softmax(out, dim=1).cpu().numpy()
                    preds = probs.argmax(axis=1)
                    for k, pid in enumerate(keep[j0:j1]):
                        pc = int(preds[k])
                        row = {"id": pid, "pred_class_idx": pc, "pred_class": int_to_class.get(pc, f"class_{pc}")}
                        for c in range(probs.shape[1]):
                            row[f"p_class{c}"] = float(probs[k, c])
                        rows.append(row)
                else:
                    vals = out.cpu().numpy()
                    for k, pid in enumerate(keep[j0:j1]):
                        rows.append({"id": pid, "pred_value": float(vals[k])})

        return pd.DataFrame(rows)

    # -------------------------
    # MODE: reps (cached reps + mask)
    # -------------------------
    # if mode == "reps":
    #     if reps_cache_pt is None:
    #         raise ValueError("mode=reps requires reps_cache_pt (use --reps-cache or meta.json).")
    #     if supervised_csv_for_alignment is None:
    #         raise ValueError(
    #             "mode=reps requires supervised_csv_for_alignment because your reps cache does not store ids. "
    #             "Provide --meta or pass --supervised-csv."
    #         )
    #
    #     reps_list, mask_list = _load_reps_cache(reps_cache_pt)
    #     sup_ids = _load_supervised_ids(supervised_csv_for_alignment)
    #
    #     if len(sup_ids) != len(reps_list) or len(sup_ids) != len(mask_list):
    #         raise ValueError(
    #             f"Alignment mismatch: supervised n={len(sup_ids)} reps n={len(reps_list)} mask n={len(mask_list)}"
    #         )
    #
    #     sup_index = {pid: i for i, pid in enumerate(sup_ids)}
    #     keep = [pid for pid in fasta_ids if pid in sup_index]
    #     if not keep:
    #         raise ValueError("No FASTA ids matched supervised ids (used to align reps cache).")
    #
    #     # infer embed dim
    #     embed_dim = int(reps_list[sup_index[keep[0]]].shape[-1])
    #
    #     model = RepsHeadModel(embed_dim=embed_dim, task=task, num_classes=num_classes).to(device_t)
    #     state = torch.load(head_checkpoint, map_location=device_t)
    #
    #     # you saved head.state_dict() from NADPHSeqModel.head, so load into model.head
    #     model.head.load_state_dict(state, strict=True)
    #     model.eval()
    #
    #     rows = []
    #     grad_enabled = (compute_saliency or compute_ig)
    #     ctx = torch.enable_grad if grad_enabled else torch.no_grad
    #
    #     with ctx():
    #         for j0 in tqdm(range(0, len(keep), batch_size), desc="Predict (reps)"):
    #             batch_ids = keep[j0:j0 + batch_size]
    #             idxs = [sup_index[pid] for pid in batch_ids]
    #
    #             reps_batch = [reps_list[i] for i in idxs]
    #             mask_batch = [mask_list[i] for i in idxs]
    #
    #             Lmax = max(int(r.shape[0]) for r in reps_batch)
    #             D = embed_dim
    #
    #             reps_pad = torch.zeros((len(batch_ids), Lmax, D), dtype=torch.float32)
    #             mask_pad = torch.zeros((len(batch_ids), Lmax), dtype=torch.bool)
    #
    #             for i, (r, m) in enumerate(zip(reps_batch, mask_batch)):
    #                 r = r.float()
    #                 m = m.bool()
    #                 L = int(r.shape[0])
    #                 reps_pad[i, :L] = r
    #                 mask_pad[i, :L] = m
    #
    #             reps_pad = reps_pad.to(device_t)
    #             mask_pad = mask_pad.to(device_t)
    #
    #             if grad_enabled:
    #                 reps_pad.requires_grad_(True)
    #
    #             logits = model(reps_pad, mask_pad)
    #
    #             if task == "classification":
    #                 probs = F.softmax(logits, dim=1)
    #                 preds = probs.argmax(dim=1)
    #                 probs_np = probs.detach().cpu().numpy()
    #                 preds_np = preds.detach().cpu().numpy()
    #             else:
    #                 preds_np = logits.detach().cpu().numpy()
    #
    #             # Explainability using your module functions (in reps-space)
    #             sal_scores = None
    #             ig_scores = None
    #
    #             if compute_saliency:
    #                 # We need a NADPHSeqModel-like object for the helper, easiest is to wrap.
    #                 # Instead, compute via your helpers by creating a dummy NADPHSeqModel with same head/pooler/norm.
    #                 dummy = NADPHSeqModel(esm_model=None, embed_dim=embed_dim, task=task, num_classes=num_classes).to(device_t)
    #                 dummy.pooler = model.pooler
    #                 dummy.norm = model.norm
    #                 dummy.head = model.head
    #                 dummy.eval()
    #                 sal_scores = compute_residue_saliency_from_reps(dummy, reps_pad, mask_pad, target_class=target_class).detach().cpu()
    #
    #             if compute_ig:
    #                 dummy = NADPHSeqModel(esm_model=None, embed_dim=embed_dim, task=task, num_classes=num_classes).to(device_t)
    #                 dummy.pooler = model.pooler
    #                 dummy.norm = model.norm
    #                 dummy.head = model.head
    #                 dummy.eval()
    #                 ig_scores = compute_residue_integrated_gradients_from_reps(
    #                     dummy, reps_pad, mask_pad, target_class=target_class, steps=ig_steps
    #                 ).detach().cpu()
    #
    #             for i, pid in enumerate(batch_ids):
    #                 row = {"id": pid}
    #                 if task == "classification":
    #                     pc = int(preds_np[i])
    #                     row["pred_class_idx"] = pc
    #                     row["pred_class"] = int_to_class.get(pc, f"class_{pc}")
    #                     for c in range(probs_np.shape[1]):
    #                         row[f"p_class{c}"] = float(probs_np[i, c])
    #                 else:
    #                     row["pred_value"] = float(preds_np[i])
    #
    #                 if compute_saliency and sal_scores is not None:
    #                     L = int(mask_pad[i].sum().item())
    #                     s = sal_scores[i, :L].numpy()
    #                     row["saliency"] = ";".join(f"{x:.6f}" for x in s)
    #
    #                 if compute_ig and ig_scores is not None:
    #                     L = int(mask_pad[i].sum().item())
    #                     g = ig_scores[i, :L].numpy()
    #                     row["ig"] = ";".join(f"{x:.6f}" for x in g)
    #
    #                 rows.append(row)
    #
    #     return pd.DataFrame(rows)

    if mode == "reps":
        if reps_cache_pt is None:
            raise ValueError("mode=reps requires reps_cache_pt (use --reps-cache or --meta).")

        ids, reps_list, mask_list = _load_reps_cache(reps_cache_pt)
        id_to_idx = {pid: i for i, pid in enumerate(ids)}

        keep = [pid for pid in fasta_ids if pid in id_to_idx]
        if not keep:
            raise ValueError("No FASTA ids matched reps cache ids.")

        embed_dim = int(reps_list[id_to_idx[keep[0]]].shape[-1])

        model = NADPHSeqModel(esm_model=None, embed_dim=embed_dim, task=task, num_classes=num_classes).to(device_t)
        state = torch.load(head_checkpoint, map_location=device_t, weights_only = False)
        model.head.load_state_dict(state, strict=True)
        model.eval()

        rows = []
        grad_enabled = (compute_saliency or compute_ig)
        ctx = torch.enable_grad if grad_enabled else torch.no_grad

        with ctx():
            for j0 in tqdm(range(0, len(keep), batch_size), desc="Predict (reps)"):
                batch_ids = keep[j0:j0 + batch_size]
                idxs = [id_to_idx[pid] for pid in batch_ids]

                reps_batch = [reps_list[i].float() for i in idxs]
                mask_batch = [mask_list[i].bool() for i in idxs]

                Lmax = max(int(r.shape[0]) for r in reps_batch)
                D = embed_dim
                B = len(batch_ids)

                reps_pad = torch.zeros((B, Lmax, D), dtype=torch.float32)
                mask_pad = torch.zeros((B, Lmax), dtype=torch.bool)
                for i, (r, m) in enumerate(zip(reps_batch, mask_batch)):
                    L = int(r.shape[0])
                    reps_pad[i, :L] = r
                    mask_pad[i, :L] = m

                reps_pad = reps_pad.to(device_t)
                mask_pad = mask_pad.to(device_t)

                logits = model(reps=reps_pad, mask=mask_pad)

                if task == "classification":
                    probs = F.softmax(logits, dim=1)
                    preds = probs.argmax(dim=1)
                    probs_np = probs.detach().cpu().numpy()
                    preds_np = preds.detach().cpu().numpy()
                else:
                    preds_np = logits.detach().cpu().numpy()

                sal_scores = None
                ig_scores = None
                if compute_saliency:
                    sal_scores = compute_residue_saliency_from_reps(model, reps_pad, mask_pad,
                                                                    target_class=target_class).cpu()
                if compute_ig:
                    ig_scores = compute_residue_integrated_gradients_from_reps(
                        model, reps_pad, mask_pad, target_class=target_class, steps=ig_steps
                    ).cpu()

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
    # MODE: tokens (runs ESM)
    # -------------------------
    esm_bs = _safe_esm_batch_size(device_t, esm_batch_size)
    print(f"Loading ESM model: {model_name} | repr_layer={repr_layer}")
    esm_model, alphabet = load_esm_model_and_alphabet(model_name)
    esm_model.to(device_t).eval()
    batch_converter = alphabet.get_batch_converter()

    embed_dim = int(esm_model.embed_dim)
    head_model = RepsHeadModel(embed_dim=embed_dim, task=task, num_classes=num_classes).to(device_t)
    state = torch.load(head_checkpoint, map_location=device_t, weights_only = False)
    head_model.head.load_state_dict(state, strict=True)
    head_model.eval()

    rows = []
    grad_enabled = (compute_saliency or compute_ig)
    ctx = torch.enable_grad if grad_enabled else torch.no_grad

    id_chunks = chunk_list(fasta_ids, esm_bs)
    print(f"Starting inference: {len(id_chunks)} batches (esm_bs={esm_bs})")

    with ctx():
        for id_chunk in tqdm(id_chunks, desc="Predict (tokens)"):
            seq_chunk: List[Tuple[str, str]] = []
            for pid in id_chunk:
                s = seq_dict[pid]
                if len(s) > max_len:
                    s = s[:max_len]
                seq_chunk.append((pid, s))

            _, _, toks = batch_converter(seq_chunk)
            toks = toks.to(device_t)
            mask = (toks != 1)

            out = esm_model(toks, repr_layers=[repr_layer], return_contacts=False)
            reps = out["representations"][repr_layer]  # (B,L,D)

            if grad_enabled:
                reps.retain_grad()

            logits = head_model(reps, mask)

            if task == "classification":
                probs = F.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                probs_np = probs.detach().cpu().numpy()
                preds_np = preds.detach().cpu().numpy()
            else:
                preds_np = logits.detach().cpu().numpy()

            # compute saliency/IG directly in reps space (fast enough per batch)
            sal_scores = None
            ig_scores = None

            if compute_saliency:
                head_model.zero_grad(set_to_none=True)
                if task == "classification":
                    if target_class is None:
                        sel = logits.gather(1, preds.unsqueeze(1)).squeeze(1)
                    else:
                        sel = logits[:, int(target_class)]
                    loss = sel.sum()
                else:
                    loss = logits.sum()
                loss.backward()
                sal_scores = (reps.grad.norm(dim=-1) * mask.float()).detach().cpu()

            if compute_ig:
                base = torch.zeros_like(reps)
                total_grad = torch.zeros_like(reps)
                for a in torch.linspace(0.0, 1.0, ig_steps, device=device_t):
                    head_model.zero_grad(set_to_none=True)
                    x = base + a * (reps.detach() - base)
                    x.requires_grad_(True)
                    o = head_model(x, mask)
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

                seq_len = len(seq_chunk[i][1])
                if compute_saliency and sal_scores is not None:
                    s = sal_scores[i, 1: seq_len + 1].numpy()
                    row["saliency"] = ";".join(f"{x:.6f}" for x in s)
                if compute_ig and ig_scores is not None:
                    g = ig_scores[i, 1: seq_len + 1].numpy()
                    row["ig"] = ";".join(f"{x:.6f}" for x in g)

                rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict NADPH responsiveness from sequences using trained head.")

    parser.add_argument("--fasta", default="../results/protein_sequences.fasta")
    parser.add_argument("--head", dest="head_checkpoint", default="../results/nadph_seq_head.pt")
    parser.add_argument("--mode", choices=["pooled", "reps", "tokens"], default="pooled")
    parser.add_argument("--meta", default=None, help="Training meta JSON (recommended).")

    parser.add_argument("--pooled-cache", default=None)
    parser.add_argument("--reps-cache", default=None)
    parser.add_argument("--supervised-csv", default=None, help="Needed to align caches if caches do not include ids.")

    parser.add_argument("--model-name", default="esm2_t33_650M_UR50D")
    parser.add_argument("--repr-layer", type=int, default=33)
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=1022)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--esm-batch-size", type=int, default=8)

    parser.add_argument("--saliency", action="store_true")
    parser.add_argument("--ig", action="store_true")
    parser.add_argument("--target-class", type=int, default=None)
    parser.add_argument("--ig-steps", type=int, default=50)

    parser.add_argument("--out", default="../results/predictions_nadph_seq.csv")

    args = parser.parse_args()

    # Fill from meta.json (preferred)
    if args.meta is not None:
        meta = json.loads(Path(args.meta).read_text())
        cfg = meta.get("config", {})
        cache_dir = cfg.get("cache_dir")
        artifacts = meta.get("artifacts", {})

        # task / model defaults
        if cfg.get("task"):
            args.task = cfg["task"]
        if cfg.get("model_name"):
            args.model_name = cfg["model_name"]
        if cfg.get("repr_layer") is not None:
            args.repr_layer = int(cfg["repr_layer"])
        if cfg.get("num_classes") is not None:
            args.num_classes = int(cfg["num_classes"])
        if cfg.get("max_len") is not None:
            args.max_len = int(cfg["max_len"])

        # supervised CSV for alignment
        if args.supervised_csv is None and artifacts.get("supervised_csv"):
            args.supervised_csv = artifacts["supervised_csv"]

        # cache paths: we reconstruct filenames using your module naming
        # because meta currently stores cache_paths as {"pooled": "...", "reps": "..."} etc.
        cache_paths = meta.get("cache_paths", {})
        if args.pooled_cache is None and cache_paths.get("pooled"):
            args.pooled_cache = cache_paths["pooled"]
        if args.reps_cache is None and cache_paths.get("reps"):
            args.reps_cache = cache_paths["reps"]

    df_pred = predict_nadph_from_fasta(
        fasta_path=args.fasta,
        head_checkpoint=args.head_checkpoint,
        mode=args.mode,
        model_name=args.model_name,
        task=args.task,
        num_classes=args.num_classes,
        max_len=args.max_len,
        repr_layer=args.repr_layer,
        device=args.device,
        batch_size=args.batch_size,
        esm_batch_size=args.esm_batch_size,
        pooled_cache_pt=args.pooled_cache,
        reps_cache_pt=args.reps_cache,
        supervised_csv_for_alignment=args.supervised_csv,
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
