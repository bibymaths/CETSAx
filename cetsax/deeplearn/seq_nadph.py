"""
seq_nadph.py
------------

Sequence-based modelling of NADPH responsiveness using protein language models (ESM2).

What this script supports (fully working):
- Build supervised table from EC50 fits + FASTA (adds seq, label_cls, label_reg).
- Cache:
  1) tokens (per protein)
  2) pooled embeddings (per protein)  -> fastest training
  3) residue representations (per protein) + masks -> explainability without rerunning ESM
- Train with three modes:
  - train_mode="pooled" : uses cached pooled embeddings (fast)
  - train_mode="reps"   : uses cached per-residue reps (fast-ish, enables explainability)
  - train_mode="tokens" : runs ESM forward during training (slowest)
- Focal Loss + class weights + WeightedRandomSampler
- Early stopping + ReduceLROnPlateau scheduler
- Saliency + Integrated Gradients from cached reps (no ESM required), and from tokens (uses ESM)

Hardware note (your setup: 1x CUDA 15GB):
- Caching pooled reps: bs auto uses 2 on CUDA, with OOM fallback to 1
- Caching residue reps: bs auto uses 1 on CUDA (safest)

BSD 3-Clause License
Copyright (c) 2025, Abhinav Mishra
Email: mishraabhinav36@gmail.com
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

import copy
import math
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler, Dataset, DataLoader
from tqdm.auto import tqdm
import esm

warnings.filterwarnings("ignore", category=UserWarning, module="torch")


# -----------------------------------------------------------------------------
# CPU threading (helps on 64 cores; avoid oversubscription elsewhere if needed)
# -----------------------------------------------------------------------------
def configure_cpu_threads(num_threads: int = 64, interop_threads: int = 4):
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    try:
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(interop_threads)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# 1) FASTA util
# -----------------------------------------------------------------------------
def read_fasta_to_dict(fasta_path: str | Path) -> Dict[str, str]:
    """
    Read FASTA into dict: {id: sequence}.
    Supports:
      >P12345
      >sp|P12345|...
    """
    fasta_path = Path(fasta_path)
    seqs: Dict[str, List[str]] = {}
    current_id: Optional[str] = None

    with fasta_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                header_token = line[1:].split()[0]
                if "|" in header_token:
                    current_id = header_token.split("|")[1]
                else:
                    current_id = header_token
                seqs.setdefault(current_id, [])
            else:
                if current_id is None:
                    continue
                seqs[current_id].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


# -----------------------------------------------------------------------------
# 2) Build supervised table from fits + FASTA
# -----------------------------------------------------------------------------
def classify_hit_row(
    row: pd.Series,
    ec50_strong: float = 0.01,
    delta_strong: float = 0.10,
    r2_strong: float = 0.70,
) -> str:
    ec50 = float(row["EC50"])
    dm = float(row["delta_max"])
    r2 = float(row["R2"])
    return "strong" if (ec50 < ec50_strong) and (dm > delta_strong) and (r2 > r2_strong) else "weak"


def build_sequence_supervised_table(
    fits_df: pd.DataFrame,
    fasta_path: str | Path,
    out_csv: str | Path,
    id_col: str = "id",
    use_nss: bool = False,
) -> pd.DataFrame:
    agg = (
        fits_df.groupby(id_col)
        .agg(EC50=("EC50", "median"), delta_max=("delta_max", "median"), R2=("R2", "median"))
        .reset_index()
    )

    agg["hit_class"] = agg.apply(classify_hit_row, axis=1)
    class_to_int = {"weak": 0, "strong": 1}
    agg["label_cls"] = agg["hit_class"].map(class_to_int).astype(int)

    if use_nss and "NSS" in agg.columns:
        agg["label_reg"] = agg["NSS"].astype(float)
    else:
        ec50 = agg["EC50"].replace(0, np.nan)
        agg["label_reg"] = -np.log10(ec50)

    # Attach sequences + rescue mismatches
    seq_dict = read_fasta_to_dict(fasta_path)
    agg["seq"] = agg[id_col].map(seq_dict)

    missing_mask = agg["seq"].isna()
    if missing_mask.any():
        clean_ids = agg.loc[missing_mask, id_col].astype(str).apply(lambda x: x.split("-")[0])
        rescued = clean_ids.map(seq_dict)
        agg.loc[missing_mask, "seq"] = rescued

    agg = agg.dropna(subset=["seq"])

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_csv, index=False)
    return agg


# -----------------------------------------------------------------------------
# 3) Config
# -----------------------------------------------------------------------------
@dataclass
class NADPHSeqConfig:
    # ESM
    model_name: str = "esm2_t33_650M_UR50D"
    repr_layer: int = 33
    max_len: int = 1022

    # task
    task: str = "classification"  # or "regression"
    num_classes: int = 2

    # training
    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 15
    patience: int = 5
    gamma_focal: float = 3.0

    # device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # caching
    cache_dir: str = "cache"
    cache_fp16: bool = True
    cache_tokens: bool = True
    cache_pooled: bool = True
    cache_reps: bool = False  # large; turn on when you want explainability at scale

    # runtime mode
    train_mode: str = "pooled"  # "pooled" | "reps" | "tokens"

    # dataloader
    head_batch_size: int = 256  # used for pooled/reps training (head-only compute)
    num_workers_pooled: int = 16
    num_workers_reps: int = 8
    num_workers_tokens: int = 4


# -----------------------------------------------------------------------------
# 4) Dataset helpers
# -----------------------------------------------------------------------------
def collate_fn_esm(batch: List[Tuple[torch.Tensor, int | float]]) -> Tuple[torch.Tensor, torch.Tensor]:
    toks, labels = zip(*batch)
    lengths = [len(t) for t in toks]
    max_len = max(lengths)
    batch_toks = torch.full((len(toks), max_len), fill_value=1, dtype=torch.long)  # 1 is ESM padding
    for i, t in enumerate(toks):
        batch_toks[i, : len(t)] = t
    labels_tensor = torch.tensor(labels, dtype=torch.long if isinstance(labels[0], int) else torch.float32)
    return batch_toks, labels_tensor


def collate_fn_reps(batch):
    reps_list, mask_list, labels = zip(*batch)
    B = len(reps_list)
    D = reps_list[0].shape[1]
    Lmax = max(r.shape[0] for r in reps_list)

    reps = torch.zeros((B, Lmax, D), dtype=torch.float32)
    mask = torch.zeros((B, Lmax), dtype=torch.bool)

    for i, (r, m) in enumerate(zip(reps_list, mask_list)):
        L = r.shape[0]
        reps[i, :L] = r
        mask[i, :L] = m

    labels_t = torch.tensor(labels, dtype=torch.long if isinstance(labels[0], int) else torch.float32)
    return reps, mask, labels_t


class NADPHSeqDataset(Dataset):
    """Tokenize from CSV on-the-fly (slowest); kept for completeness."""
    def __init__(self, csv_path: str | Path, alphabet, task="classification", max_len=1022):
        self.df = pd.read_csv(csv_path)
        self.batch_converter = alphabet.get_batch_converter()
        self.task = task
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq = str(row["seq"])[: self.max_len]
        seq_id = str(row["id"])
        _, _, toks = self.batch_converter([(seq_id, seq)])
        toks = toks[0]
        if self.task == "classification":
            return toks, int(row["label_cls"])
        return toks, float(row["label_reg"])


class NADPHSeqDatasetTokenCached(Dataset):
    """Load tokenized sequences from token cache."""
    def __init__(self, token_cache_pt: str | Path, task="classification"):
        obj = torch.load(token_cache_pt, map_location="cpu")
        self.tokens = obj["tokens"]
        self.task = task
        self.labels = obj["label_cls"] if task == "classification" else obj["label_reg"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        t = self.tokens[idx]
        if self.task == "classification":
            return t, int(self.labels[idx])
        return t, float(self.labels[idx])


class NADPHPooledDataset(Dataset):
    """Load cached pooled embeddings (N, D)."""
    def __init__(self, pooled_pt: str | Path, task="classification"):
        obj = torch.load(pooled_pt, map_location="cpu")
        self.x = obj["pooled"]
        self.task = task
        self.y = obj["label_cls"].long() if task == "classification" else obj["label_reg"].float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i].float()
        if self.task == "classification":
            return x, int(self.y[i])
        return x, float(self.y[i])


class NADPHRepsDataset(Dataset):
    """Load cached residue reps (list of (L,D)) + masks (list of (L,))."""
    def __init__(self, reps_pt: str | Path, task="classification"):
        obj = torch.load(reps_pt, map_location="cpu")
        self.reps = obj["reps"]
        self.mask = obj["mask"]
        self.task = task
        self.y = obj["label_cls"].long() if task == "classification" else obj["label_reg"].float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        reps = self.reps[i].float()
        mask = self.mask[i].bool()
        if self.task == "classification":
            return reps, mask, int(self.y[i])
        return reps, mask, float(self.y[i])


# -----------------------------------------------------------------------------
# 5) Losses + pooling + model
# -----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal loss for multi-class classification."""
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        fl = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return fl.mean()
        if self.reduction == "sum":
            return fl.sum()
        return fl


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention_weights = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B,L,D), mask: (B,L) bool
        scores = self.attention_weights(x).squeeze(-1)  # (B,L)
        scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=1)
        weighted = torch.sum(attn.unsqueeze(-1) * x, dim=1)  # (B,D)
        return weighted


class NADPHSeqModel(nn.Module):
    """
    Single model that supports:
      - tokens path (runs ESM)
      - reps path (cached per-residue reps)
      - pooled path (cached pooled embeddings)
    """
    def __init__(self, esm_model, embed_dim: int, task="classification", num_classes=2, dropout=0.3):
        super().__init__()
        self.esm = esm_model
        self.task = task

        if self.esm is not None:
            for p in self.esm.parameters():
                p.requires_grad = False

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

    def forward(
        self,
        tokens: torch.Tensor | None = None,
        reps: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pooled: torch.Tensor | None = None,
        return_reps: bool = False,
        repr_layer: int = 33,
    ):
        # (A) pooled path
        if pooled is not None:
            x = self.norm(pooled)
            logits = self.head(x)
            if self.task == "regression":
                logits = logits.squeeze(-1)
            return logits if not return_reps else (logits, None, None)

        # (B) cached reps path
        if reps is not None and mask is not None:
            pooled2 = self.pooler(reps, mask)
            pooled2 = self.norm(pooled2)
            logits = self.head(pooled2)
            if self.task == "regression":
                logits = logits.squeeze(-1)
            return logits if not return_reps else (logits, reps, mask)

        # (C) tokens path
        assert tokens is not None, "Provide pooled, reps+mask, or tokens."
        assert self.esm is not None, "ESM model is required for tokens path."
        mask3 = (tokens != 1)
        with torch.no_grad():
            out = self.esm(tokens, repr_layers=[repr_layer], return_contacts=False)
            reps3 = out["representations"][repr_layer]
        pooled3 = self.pooler(reps3, mask3)
        pooled3 = self.norm(pooled3)
        logits = self.head(pooled3)
        if self.task == "regression":
            logits = logits.squeeze(-1)
        return logits if not return_reps else (logits, reps3, mask3)


# -----------------------------------------------------------------------------
# 6) Explainability (works from cached reps or from tokens)
# -----------------------------------------------------------------------------
def compute_residue_saliency_from_reps(
    model: NADPHSeqModel,
    reps: torch.Tensor,
    mask: torch.Tensor,
    target_class: int | None = None,
) -> torch.Tensor:
    """
    reps: (B,L,D) float32 requires_grad
    mask: (B,L) bool
    returns: (B,L) saliency scores
    """
    model.zero_grad(set_to_none=True)
    reps = reps.clone().detach().requires_grad_(True)
    logits, reps_out, mask_out = model(reps=reps, mask=mask, return_reps=True)

    if model.task == "classification":
        if target_class is None:
            pred = logits.argmax(dim=1)
        else:
            pred = torch.full((logits.size(0),), int(target_class), device=logits.device, dtype=torch.long)
        selected = logits.gather(1, pred.unsqueeze(1)).squeeze(1)
        loss = selected.sum()
    else:
        loss = logits.sum()

    loss.backward()
    sal = reps.grad.norm(dim=-1) * mask.float()
    return sal


def compute_residue_integrated_gradients_from_reps(
    model: NADPHSeqModel,
    reps: torch.Tensor,
    mask: torch.Tensor,
    target_class: int | None = None,
    steps: int = 50,
) -> torch.Tensor:
    """
    Integrated gradients in representation space.
    reps: (B,L,D)
    mask: (B,L)
    returns: (B,L) IG scores
    """
    device = next(model.parameters()).device
    reps = reps.to(device)
    mask = mask.to(device)

    baseline = torch.zeros_like(reps)
    total_grad = torch.zeros_like(reps)

    # choose the class to explain based on model output at the true point
    with torch.no_grad():
        logits0 = model(reps=reps, mask=mask)
        if model.task == "classification":
            if target_class is None:
                base_pred = logits0.argmax(dim=1)
            else:
                base_pred = torch.full((logits0.size(0),), int(target_class), device=device, dtype=torch.long)
        else:
            base_pred = None

    for a in torch.linspace(0.0, 1.0, steps, device=device):
        model.zero_grad(set_to_none=True)
        reps_i = (baseline + a * (reps - baseline)).detach().requires_grad_(True)

        logits_i = model(reps=reps_i, mask=mask)
        if model.task == "classification":
            selected = logits_i.gather(1, base_pred.unsqueeze(1)).squeeze(1)
        else:
            selected = logits_i
        loss = selected.sum()
        loss.backward()
        total_grad += reps_i.grad

    avg_grad = total_grad / float(steps)
    ig = (reps - baseline) * avg_grad
    ig_scores = ig.norm(dim=-1) * mask.float()
    return ig_scores


# -----------------------------------------------------------------------------
# 7) Caching
# -----------------------------------------------------------------------------
def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_esm_model_and_alphabet(model_name: str):
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    return model, alphabet


def build_token_cache(csv_path: str | Path, cfg: NADPHSeqConfig) -> Path:
    """
    Tokenize sequences once and save {tokens: List[Tensor(L)], label_cls, label_reg?}.
    """
    cache_dir = _ensure_dir(cfg.cache_dir)
    out_pt = cache_dir / f"tokens_{Path(csv_path).stem}_{cfg.max_len}.pt"
    if out_pt.exists():
        return out_pt

    _, alphabet = load_esm_model_and_alphabet(cfg.model_name)
    batch_converter = alphabet.get_batch_converter()

    df = pd.read_csv(csv_path)

    tokens: list[torch.Tensor] = []
    for rid, seq in zip(df["id"].astype(str), df["seq"].astype(str)):
        seq = seq[: cfg.max_len]
        _, _, toks = batch_converter([(rid, seq)])
        tokens.append(toks[0].cpu())

    labels_cls = torch.tensor(df["label_cls"].values, dtype=torch.long)
    payload: dict[str, Any] = {"tokens": tokens, "label_cls": labels_cls}

    if "label_reg" in df.columns:
        payload["label_reg"] = torch.tensor(df["label_reg"].values, dtype=torch.float32)

    torch.save(payload, out_pt)
    print(f"[cache] wrote token cache: {out_pt} (n={len(tokens)})")
    return out_pt


@torch.no_grad()
def build_pooled_cache(token_cache_pt: str | Path, cfg: NADPHSeqConfig) -> Path:
    """
    Compute pooled (D,) embedding per protein using ESM reps + AttentionPooling.
    Saves: {"pooled": (N,D), "label_cls":..., "label_reg"?:...}
    """
    cache_dir = _ensure_dir(cfg.cache_dir)
    out_pt = cache_dir / f"pooled_{Path(token_cache_pt).stem}_{cfg.model_name}_L{cfg.repr_layer}.pt"
    if out_pt.exists():
        return out_pt

    device = torch.device(cfg.device)
    esm_model, _ = load_esm_model_and_alphabet(cfg.model_name)
    esm_model.to(device).eval()

    obj = torch.load(token_cache_pt, map_location="cpu")
    tokens_list = obj["tokens"]
    y_cls = obj["label_cls"]
    y_reg = obj.get("label_reg", None)

    embed_dim = esm_model.embed_dim
    pooler = AttentionPooling(embed_dim).to(device).eval()
    norm = nn.LayerNorm(embed_dim).to(device).eval()

    dtype = torch.float16 if cfg.cache_fp16 else torch.float32
    pooled_all = torch.empty((len(tokens_list), embed_dim), dtype=dtype)

    # Safe defaults for 1x15GB GPU
    bs = 2 if device.type == "cuda" else max(1, cfg.batch_size)

    def _run_batch(toks_batch: torch.Tensor) -> torch.Tensor:
        mask = (toks_batch != 1)
        out = esm_model(toks_batch, repr_layers=[cfg.repr_layer], return_contacts=False)
        reps = out["representations"][cfg.repr_layer]  # (B,L,D)
        pooled = pooler(reps, mask)
        pooled = norm(pooled)
        return pooled

    i = 0
    pbar = tqdm(total=len(tokens_list), desc="Caching pooled embeddings")
    while i < len(tokens_list):
        b = min(bs, len(tokens_list) - i)
        batch_tokens = tokens_list[i : i + b]
        toks = collate_fn_esm([(t, 0) for t in batch_tokens])[0].to(device)
        try:
            pooled = _run_batch(toks).to("cpu")
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda" and b > 1:
                torch.cuda.empty_cache()
                bs = 1
                continue
            raise

        if dtype == torch.float16:
            pooled = pooled.half()
        pooled_all[i : i + pooled.size(0)] = pooled
        i += pooled.size(0)
        pbar.update(pooled.size(0))
    pbar.close()

    payload: dict[str, Any] = {"pooled": pooled_all, "label_cls": y_cls}
    if y_reg is not None:
        payload["label_reg"] = y_reg

    torch.save(payload, out_pt)
    print(f"[cache] wrote pooled cache: {out_pt} shape={tuple(pooled_all.shape)} dtype={pooled_all.dtype}")
    return out_pt


@torch.no_grad()
def build_reps_cache(token_cache_pt: str | Path, cfg: NADPHSeqConfig) -> Path:
    """
    Cache per-residue reps so saliency/IG can run without ESM forward.
    Saves:
      {"reps": List[(L,D)], "mask": List[(L,)], "label_cls":..., "label_reg"?:...}
    """
    cache_dir = _ensure_dir(cfg.cache_dir)
    out_pt = cache_dir / f"reps_{Path(token_cache_pt).stem}_{cfg.model_name}_L{cfg.repr_layer}.pt"
    if out_pt.exists():
        return out_pt

    device = torch.device(cfg.device)
    esm_model, _ = load_esm_model_and_alphabet(cfg.model_name)
    esm_model.to(device).eval()

    obj = torch.load(token_cache_pt, map_location="cpu")
    tokens_list = obj["tokens"]
    y_cls = obj["label_cls"]
    y_reg = obj.get("label_reg", None)

    dtype = torch.float16 if cfg.cache_fp16 else torch.float32
    reps_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor] = []

    # safest for 15GB (variable long sequences)
    bs = 1 if device.type == "cuda" else max(1, cfg.batch_size)

    pbar = tqdm(total=len(tokens_list), desc="Caching residue reps")
    for i in range(0, len(tokens_list), bs):
        batch_tokens = tokens_list[i : i + bs]
        toks = collate_fn_esm([(t, 0) for t in batch_tokens])[0].to(device)
        mask = (toks != 1)

        out = esm_model(toks, repr_layers=[cfg.repr_layer], return_contacts=False)
        reps = out["representations"][cfg.repr_layer]  # (B,L,D)

        reps = reps.to("cpu")
        mask = mask.to("cpu")

        if dtype == torch.float16:
            reps = reps.half()

        for b in range(reps.size(0)):
            L = int(mask[b].sum().item())
            reps_list.append(reps[b, :L].contiguous())      # (L,D)
            mask_list.append(mask[b, :L].contiguous())      # (L,)
        pbar.update(len(batch_tokens))
    pbar.close()

    payload: dict[str, Any] = {"reps": reps_list, "mask": mask_list, "label_cls": y_cls}
    if y_reg is not None:
        payload["label_reg"] = y_reg

    torch.save(payload, out_pt)
    print(f"[cache] wrote reps cache: {out_pt} n={len(reps_list)} dtype={dtype}")
    return out_pt


# -----------------------------------------------------------------------------
# 8) Training
# -----------------------------------------------------------------------------

def _get_head_module(model: nn.Module) -> nn.Module:
    """
    Return the trainable head module regardless of model type:
    - NADPHSeqModel: model.head
    - HeadOnlyModel: model.net
    - DataParallel/DDP: unwrap .module
    """
    base = model.module if hasattr(model, "module") else model

    if hasattr(base, "head") and isinstance(getattr(base, "head"), nn.Module):
        return base.head

    if hasattr(base, "net") and isinstance(getattr(base, "net"), nn.Module):
        return base.net

    raise AttributeError(f"Could not find head on model type={type(base)} (expected .head or .net)")

def _split_train_val(ds: Dataset, seed: int = 0, train_frac: float = 0.8):
    n = len(ds)
    n_train = int(train_frac * n)
    n_val = n - n_train
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(ds, [n_train, n_val], generator=g)


def _get_labels_for_split(split_ds) -> List[int] | List[float]:
    """
    Efficiently extract labels from a random_split subset when possible.
    Falls back to indexing.
    """
    base = getattr(split_ds, "dataset", None)
    idxs = getattr(split_ds, "indices", None)

    # pooled dataset: y exists on base
    if base is not None and idxs is not None and hasattr(base, "y"):
        y = base.y[idxs]
        return y.tolist()

    # reps dataset: y exists on base
    if base is not None and idxs is not None and hasattr(base, "y"):
        y = base.y[idxs]
        return y.tolist()

    # tokens datasets: compute by indexing
    labels = []
    for i in range(len(split_ds)):
        item = split_ds[i]
        labels.append(item[-1] if isinstance(item, tuple) else item[1])
    return labels


def train_seq_model(
    csv_path: str | Path,
    cfg: NADPHSeqConfig,
    patience: bool = True,
) -> Tuple[nn.Module, Dict[str, float], Dict[str, Path], pd.DataFrame]:
    """
    Returns:
      model, metrics, paths(dict of caches)
    """
    configure_cpu_threads(num_threads=64, interop_threads=4)
    device = torch.device(cfg.device)

    cache_paths: Dict[str, Path] = {}

    # ---- build caches ----
    token_cache_pt = None
    if cfg.cache_tokens or cfg.cache_pooled or cfg.cache_reps or cfg.train_mode in ("pooled", "reps", "tokens"):
        token_cache_pt = build_token_cache(csv_path, cfg)
        cache_paths["tokens"] = token_cache_pt

    pooled_pt = None
    if cfg.cache_pooled or cfg.train_mode == "pooled":
        pooled_pt = build_pooled_cache(token_cache_pt, cfg)
        cache_paths["pooled"] = pooled_pt

    reps_pt = None
    if cfg.cache_reps or cfg.train_mode == "reps":
        reps_pt = build_reps_cache(token_cache_pt, cfg)
        cache_paths["reps"] = reps_pt

    # ---- dataset + collate based on mode ----
    mode = cfg.train_mode.lower().strip()
    if mode == "pooled":
        ds = NADPHPooledDataset(pooled_pt, task=cfg.task)
        collate = None
        batch_size = cfg.head_batch_size
        n_workers = cfg.num_workers_pooled
    elif mode == "reps":
        ds = NADPHRepsDataset(reps_pt, task=cfg.task)
        collate = collate_fn_reps
        batch_size = max(8, min(64, cfg.head_batch_size // 4))
        n_workers = cfg.num_workers_reps
    elif mode == "tokens":
        ds = NADPHSeqDatasetTokenCached(token_cache_pt, task=cfg.task)
        collate = collate_fn_esm
        batch_size = cfg.batch_size
        n_workers = cfg.num_workers_tokens
    else:
        raise ValueError(f"Unknown train_mode={cfg.train_mode} (use pooled|reps|tokens)")

    train_ds, val_ds = _split_train_val(ds, seed=0, train_frac=0.8)

    # ---- model: we keep NADPHSeqModel always, but ESM is only needed for tokens path ----
    esm_model = None
    embed_dim = None

    if mode == "tokens":
        esm_model, _ = load_esm_model_and_alphabet(cfg.model_name)
        esm_model.to(device).eval()
        embed_dim = esm_model.embed_dim
    else:
        # no ESM needed during training; we only need embed_dim to build the head
        # cheapest safe way: load model on CPU once
        tmp_model, _ = load_esm_model_and_alphabet(cfg.model_name)
        embed_dim = tmp_model.embed_dim
        esm_model = None  # not used in pooled/reps training

    model = NADPHSeqModel(
        esm_model=esm_model,
        embed_dim=embed_dim,
        task=cfg.task,
        num_classes=cfg.num_classes,
        dropout=0.3,
    ).to(device)

    # ---- loss + sampler ----
    sampler_train = None
    sampler_val = None

    if cfg.task == "classification":
        train_labels = _get_labels_for_split(train_ds)
        val_labels = _get_labels_for_split(val_ds)

        class_counts = torch.bincount(torch.tensor(train_labels, dtype=torch.long)).float()
        class_counts[class_counts == 0] = 1.0

        weights = 1.0 / torch.sqrt(class_counts)
        weights = weights / weights.sum() * cfg.num_classes
        weights_cpu = weights.detach().cpu()
        weights_dev = weights_cpu.to(device)

        criterion = FocalLoss(alpha=weights_dev, gamma=cfg.gamma_focal)

        sample_weights_train = [float(weights_cpu[int(lbl)]) for lbl in train_labels]
        sample_weights_val = [float(weights_cpu[int(lbl)]) for lbl in val_labels]

        sampler_train = WeightedRandomSampler(sample_weights_train, num_samples=len(train_ds), replacement=True)
        sampler_val = WeightedRandomSampler(sample_weights_val, num_samples=len(val_ds), replacement=True)
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=cfg.lr)

    # ReduceLROnPlateau is OK for both tasks (monitor val_loss)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, cfg.patience)
    )

    pin_memory = (device.type == "cuda")
    persistent_workers = True if n_workers > 0 else False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler_train,
        shuffle=False if sampler_train is not None else True,
        collate_fn=collate,
        num_workers=n_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=sampler_val,
        shuffle=False,
        collate_fn=collate,
        num_workers=n_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )

    # ---- run ----
    run_info = {
        "csv_path": str(csv_path),
        "task": str(cfg.task),
        "train_mode": str(cfg.train_mode),
        "model_name": str(cfg.model_name),
        "repr_layer": int(cfg.repr_layer),
        "max_len": int(cfg.max_len),
        "device": str(device),
        "batch_size": int(batch_size),
        "head_batch_size": int(cfg.head_batch_size),
        "lr_init": float(cfg.lr),
        "epochs_cfg": int(cfg.epochs),
        "patience_cfg": int(cfg.patience),
        "gamma_focal": float(cfg.gamma_focal) if cfg.task == "classification" else None,
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds)),
    }
    history_rows: list[dict] = []

    # ---- early stopping ----
    best_model_state = None
    best_val_loss_monitor = math.inf
    best_val_acc = -math.inf
    trigger_times = 0

    metrics: Dict[str, float] = {}

    for epoch in range(1, cfg.epochs + 1):
        # TRAIN
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            if mode == "pooled":
                x, labels = batch
                x = x.to(device)
                labels = labels.to(device)
                outputs = model(pooled=x, repr_layer=cfg.repr_layer)
            elif mode == "reps":
                reps, mask, labels = batch
                reps = reps.to(device)
                mask = mask.to(device)
                labels = labels.to(device)
                outputs = model(reps=reps, mask=mask, repr_layer=cfg.repr_layer)
            else:
                toks, labels = batch
                toks = toks.to(device)
                labels = labels.to(device)
                outputs = model(tokens=toks, repr_layer=cfg.repr_layer)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())
            n_train_batches += 1

        train_loss /= max(1, n_train_batches)

        # VALIDATION
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                if mode == "pooled":
                    x, labels = batch
                    x = x.to(device)
                    labels = labels.to(device)
                    outputs = model(pooled=x, repr_layer=cfg.repr_layer)
                elif mode == "reps":
                    reps, mask, labels = batch
                    reps = reps.to(device)
                    mask = mask.to(device)
                    labels = labels.to(device)
                    outputs = model(reps=reps, mask=mask, repr_layer=cfg.repr_layer)
                else:
                    toks, labels = batch
                    toks = toks.to(device)
                    labels = labels.to(device)
                    outputs = model(tokens=toks, repr_layer=cfg.repr_layer)

                loss = criterion(outputs, labels)
                val_loss += float(loss.item())
                n_val_batches += 1

                if cfg.task == "classification":
                    preds = outputs.argmax(dim=1)
                    correct += int((preds == labels).sum().item())
                    total += int(labels.size(0))

        val_loss /= max(1, n_val_batches)
        scheduler.step(val_loss)

        lr_now = float(optimizer.param_groups[0]["lr"])

        if cfg.task == "classification":
            val_acc = correct / max(1, total)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            best_val_acc = max(best_val_acc, val_acc)
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

            # ---- append epoch row (DataFrame) ----
        history_rows.append(
            {
                **run_info,
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_acc": (float(val_acc) if val_acc is not None else None),
                "lr": float(lr_now),
                "early_stop_trigger_times": int(trigger_times),
                "best_val_loss_so_far": float(min(best_val_loss_monitor, val_loss)),
            }
        )

        if patience:
            if val_loss < best_val_loss_monitor:
                best_val_loss_monitor = val_loss
                head_mod = _get_head_module(model)
                best_model_state = copy.deepcopy(head_mod.state_dict())
                trigger_times = 0
            else:
                trigger_times += 1
                print(f"  >> No improvement in val_loss. Patience: {trigger_times}/{cfg.patience}")
                if trigger_times >= cfg.patience:
                    print("  >> Early stopping triggered!")
                    break

    if best_model_state is not None:
        head_mod = _get_head_module(model)
        head_mod.load_state_dict(best_model_state)
        print(f"Restored best head from memory (val_loss={best_val_loss_monitor:.4f})")

    metrics["best_val_loss"] = float(best_val_loss_monitor)
    if cfg.task == "classification":
        metrics["best_val_acc"] = float(best_val_acc)

    history_df = pd.DataFrame(history_rows)

    return model, metrics, cache_paths, history_df