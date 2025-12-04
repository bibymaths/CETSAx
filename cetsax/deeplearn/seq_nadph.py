"""
seq_nadph.py
------------

Sequence-based modelling of NADPH responsiveness using protein language models.
This module provides utilities to build supervised training tables
from CETSA NADPH response measurements and protein sequences,
and defines a PyTorch dataset and model architecture
for training deep learning models using ESM embeddings.

UPDATES:
- Implemented Focal Loss to handle hard-to-classify Strong hits.
- Integrated Class Weighting to handle dataset imbalance.
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

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import copy
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import esm
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='torch')


# ---------------------------------------------------------------------
# 1. Utility: read FASTA as id -> sequence
# ---------------------------------------------------------------------

def read_fasta_to_dict(fasta_path: str | Path) -> Dict[str, str]:
    """
    Read a FASTA file into a dict: {id: sequence}.
    Handles both simple headers (>P12345) and UniProt headers (>sp|P12345|...).
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

                if current_id not in seqs:
                    seqs[current_id] = []
            else:
                if current_id is None:
                    continue
                seqs[current_id].append(line)

    return {k: "".join(v) for k, v in seqs.items()}


# ---------------------------------------------------------------------
# 2. Build supervised table from ec50_fits + FASTA
# ---------------------------------------------------------------------

def classify_hit_row(
        row: pd.Series,
        ec50_strong: float = 0.01,
        delta_strong: float = 0.10,
        r2_strong: float = 0.70,
) -> str:
    ec50 = float(row["EC50"])
    dm = float(row["delta_max"])
    r2 = float(row["R2"])

    if (ec50 < ec50_strong) and (dm > delta_strong) and (r2 > r2_strong):
        return "strong"
    else:
        return "weak"


def build_sequence_supervised_table(
        fits_df: pd.DataFrame,
        fasta_path: str | Path,
        out_csv: str | Path,
        id_col: str = "id",
        use_nss: bool = False,
) -> pd.DataFrame:
    agg = (
        fits_df.groupby(id_col)
        .agg(
            EC50=("EC50", "median"),
            delta_max=("delta_max", "median"),
            R2=("R2", "median"),
        )
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

        # ---------------------------------------------------------
        # DIAGNOSTICS & RESCUE SECTION
        # ---------------------------------------------------------
        seq_dict = read_fasta_to_dict(fasta_path)
        csv_ids = set(agg[id_col])
        fasta_ids = set(seq_dict.keys())

        print("\n--- DEBUGGING ID MISMATCH ---")
        print(f"Total IDs in Table: {len(csv_ids)}")
        print(f"Total IDs in FASTA: {len(fasta_ids)}")

        common = csv_ids.intersection(fasta_ids)
        print(f"Direct Matches: {len(common)}")

        agg["seq"] = agg[id_col].map(seq_dict)

        missing_mask = agg["seq"].isna()
        n_missing = missing_mask.sum()

        if n_missing > 0:
            print(f"Missing Sequences: {n_missing}")
            clean_ids = agg.loc[missing_mask, id_col].astype(str).apply(lambda x: x.split('-')[0])
            rescued_seqs = clean_ids.map(seq_dict)
            agg.loc[missing_mask, "seq"] = rescued_seqs
            n_still_missing = agg["seq"].isna().sum()
            print(f"Still Missing: {n_still_missing}")

        agg = agg.dropna(subset=["seq"])
        # ---------------------------------------------------------

        out_csv = Path(out_csv)
        agg.to_csv(out_csv, index=False)
        return agg

# ---------------------------------------------------------------------
# 3. ESM-based Dataset
# ---------------------------------------------------------------------

@dataclass
class NADPHSeqConfig:
    model_name: str = "esm2_t33_650M_UR50D"
    max_len: int = 1022
    task: str = "classification"  # or "regression"
    num_classes: int = 3
    batch_size: int = 8
    lr: float = 1e-4  # Lowered slightly for Focal Loss stability
    epochs: int = 15   # Increased slightly for better convergence
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class NADPHSeqDataset(Dataset):
    def __init__(
            self,
            csv_path: str | Path,
            alphabet,
            task: str = "classification",
            max_len: int = 1022,
    ):
        self.df = pd.read_csv(csv_path)
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.task = task
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq = str(row["seq"])
        seq_id = str(row["id"])

        if len(seq) > self.max_len:
            seq = seq[: self.max_len]

        _, _, toks = self.batch_converter([(seq_id, seq)])
        toks = toks[0]

        if self.task == "classification":
            label = int(row["label_cls"])
            return toks, label
        else:
            label = float(row["label_reg"])
            return toks, label


def collate_fn_esm(batch: List[Tuple[torch.Tensor, int | float]]) -> Tuple[torch.Tensor, torch.Tensor]:
    toks, labels = zip(*batch)
    lengths = [len(t) for t in toks]
    max_len = max(lengths)
    batch_toks = torch.full(
        (len(toks), max_len), fill_value=1, dtype=torch.long
    )
    for i, t in enumerate(toks):
        batch_toks[i, : len(t)] = t
    labels_tensor = torch.tensor(labels, dtype=torch.long if isinstance(labels[0], int) else torch.float32)
    return batch_toks, labels_tensor


# ---------------------------------------------------------------------
# 4. Model: frozen ESM encoder + MLP head (+ Focal Loss)
# ---------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    This reduces the relative loss for well-classified examples (p_t > 0.5)
    and puts more focus on hard, misclassified examples.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (Tensor, optional): Weights for each class.
            gamma (float): Focusing parameter (default 2.0).
            reduction (str): 'mean' or 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (B, C) logits
        # targets: (B) labels
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # probability of the true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention_weights = nn.Linear(embed_dim, 1)

    def forward(self, x, mask):
        scores = self.attention_weights(x).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=1)
        weighted = torch.sum(attn.unsqueeze(-1) * x, dim=1)
        return weighted


class NADPHSeqModel(nn.Module):
    def __init__(
            self,
            esm_model,
            embed_dim: int,
            task: str = "classification",
            num_classes: int = 2,
            dropout: float = 0.3, # Increased dropout slightly
    ):
        super().__init__()
        self.esm = esm_model
        self.task = task

        for p in self.esm.parameters():
            p.requires_grad = False

        for n, p in self.esm.named_parameters():
            if "layers.32" in n:
                p.requires_grad = True

        self.pooler = AttentionPooling(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        hidden = 256
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes if task == "classification" else 1),
        )

    def forward(self, tokens, return_reps=False):
        mask = (tokens != 1)
        out = self.esm(tokens, repr_layers=[33], return_contacts=False)
        reps = out["representations"][33]
        pooled = self.pooler(reps, mask)
        pooled = self.norm(pooled)
        logits = self.head(pooled)

        if self.task == "regression":
            logits = logits.squeeze(-1)

        if return_reps:
            return logits, reps, mask
        return logits


def compute_residue_saliency(
        model: NADPHSeqModel,
        tokens: torch.Tensor,
        target_class: int | None = None,
) -> torch.Tensor:
    model.zero_grad()
    logits, reps, mask = model(tokens, return_reps=True)
    reps.retain_grad()

    if model.task == "classification":
        if target_class is None:
            pred_cls = logits.argmax(dim=1)
        else:
            pred_cls = torch.full((logits.size(0),), fill_value=int(target_class), dtype=torch.long, device=logits.device)
        selected = logits.gather(1, pred_cls.unsqueeze(1)).squeeze(1)
        loss = selected.sum()
    else:
        loss = logits.sum()

    loss.backward()
    grad = reps.grad
    sal = grad.norm(dim=-1)
    sal = sal * mask.float()
    return sal


def compute_residue_integrated_gradients(
        model: NADPHSeqModel,
        tokens: torch.Tensor,
        target_class: int | None = None,
        steps: int = 50,
) -> torch.Tensor:
    device = next(model.parameters()).device
    model.zero_grad()
    logits, reps, mask = model(tokens.to(device), return_reps=True)
    reps_detached = reps.detach()
    mask = mask.to(device)
    baseline = torch.zeros_like(reps_detached)
    total_grad = torch.zeros_like(reps_detached)

    for alpha in torch.linspace(0.0, 1.0, steps, device=device):
        model.zero_grad()
        reps_interp = baseline + alpha * (reps_detached - baseline)
        reps_interp.requires_grad_(True)

        mask_exp = mask.unsqueeze(-1)
        reps_masked = reps_interp * mask_exp
        lengths = mask_exp.sum(dim=1)
        pooled = reps_masked.sum(dim=1) / lengths.clamp(min=1)

        logits_interp = model.head(pooled)
        if model.task == "regression":
            selected = logits_interp
        else:
            if target_class is None:
                base_pred = logits.argmax(dim=1)
                selected = logits_interp.gather(1, base_pred.unsqueeze(1)).squeeze(1)
            else:
                cls_idx = int(target_class)
                selected = logits_interp[:, cls_idx]

        loss = selected.sum()
        loss.backward()
        grad = reps_interp.grad
        total_grad += grad

    avg_grad = total_grad / float(steps)
    ig = (reps_detached - baseline) * avg_grad
    ig_scores = ig.norm(dim=-1) * mask.float()
    return ig_scores


# ---------------------------------------------------------------------
# 5. Training utilities
# ---------------------------------------------------------------------

def load_esm_model_and_alphabet(model_name: str = "esm2_t33_650M_UR50D"):
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    return model, alphabet


def train_seq_model(
        csv_path: str | Path,
        cfg: NADPHSeqConfig,
) -> Tuple[NADPHSeqModel, Dict[str, float]]:
    """
    Train a sequence-based NADPH responsiveness model.
    """
    global sampler_train, sampler_val
    device = torch.device(cfg.device)

    # 1. Load ESM model
    esm_model, alphabet = load_esm_model_and_alphabet(cfg.model_name)
    esm_model.to(device)

    # 2. Prepare Dataset
    ds = NADPHSeqDataset(csv_path, alphabet, task=cfg.task, max_len=cfg.max_len)

    # Simple train/val split
    n = len(ds)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    n_workers = 32 if torch.cuda.is_available() else 0
    pin_memory = True if device.type == 'cuda' else False

    # 3. Initialize Model
    embed_dim = esm_model.embed_dim
    model = NADPHSeqModel(
        esm_model,
        embed_dim=embed_dim,
        task=cfg.task,
        num_classes=cfg.num_classes,
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # 4. Calculate Class Weights & Init Focal Loss
    if cfg.task == "classification":
        print("Calculating class weights for Focal Loss...")
        train_labels = [label for _, label in train_ds]
        class_counts = torch.bincount(torch.tensor(train_labels))

        class_counts = class_counts.float()
        class_counts[class_counts == 0] = 1.0

        # Calculate DAMPENED Weights (Square Root method)
        # This prevents the model from over-predicting the minority class
        weights = 1.0 / torch.sqrt(class_counts)
        weights = weights / weights.sum() * cfg.num_classes
        weights = weights.to(device)

        print(f"Class counts: {class_counts}")
        print(f"Class weights (alpha): {weights}")

        # USE FOCAL LOSS HERE
        # gamma=2.0 is standard for hard mining
        criterion = FocalLoss(alpha=weights, gamma=3.0)

        # 1. Calculate weights for each SAMPLE (not just class)
        sample_weights_train = [weights[label] for _, label in train_ds]
        sample_weights_train = torch.stack(sample_weights_train)

        sample_weights_val = [weights[label] for _, label in val_ds]
        sample_weights_val = torch.stack(sample_weights_val)

        # 2. Create the sampler
        sampler_train = WeightedRandomSampler(
            weights=sample_weights_train,
            num_samples=len(train_ds),
            replacement=True
        )

        sampler_val = WeightedRandomSampler(
            weights=sample_weights_val,
            num_samples=len(val_ds),
            replacement=True
        )

    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=cfg.lr)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler_train,
        # shuffle=True,
        collate_fn=collate_fn_esm,
        num_workers=n_workers,
        pin_memory=pin_memory,
        persistent_workers=True if n_workers> 0 else False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        sampler=sampler_val,
        # shuffle=True,
        collate_fn=collate_fn_esm,
        num_workers=n_workers,
        pin_memory=pin_memory,
        persistent_workers=True if n_workers > 0 else False,
        drop_last=True
    )

    # Optional: Scheduler to reduce LR on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 5. Setup Early Stopping
    best_val_acc = -math.inf
    best_val_loss = math.inf
    best_val_loss_monitor = math.inf

    patience = 5 # Increased patience slightly for Focal Loss
    trigger_times = 0
    best_model_state = None

    metrics = {}

    # 6. Training Loop
    for epoch in range(1, cfg.epochs + 1):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for toks, labels in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            toks = toks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(toks)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

        train_loss /= max(n_train_batches, 1)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for toks, labels in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                toks = toks.to(device)
                labels = labels.to(device)
                outputs = model(toks)

                # Validation loss should also use Focal Loss to track improvement correctly
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                n_val_batches += 1

                if cfg.task == "classification":
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

        val_loss /= max(n_val_batches, 1)

        # Step the scheduler
        scheduler.step(val_loss)

        # --- LOGGING & TRACKING ---
        if cfg.task == "classification":
            val_acc = correct / max(total, 1)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        # --- EARLY STOPPING LOGIC ---
        if val_loss < best_val_loss_monitor:
            best_val_loss_monitor = val_loss
            trigger_times = 0
            best_model_state = copy.deepcopy(model.head.state_dict())
        else:
            trigger_times += 1
            print(f"  >> No improvement in val_loss. Patience: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("  >> Early stopping triggered!")
                break

    # 7. Restore Best Model
    if best_model_state is not None:
        model.head.load_state_dict(best_model_state)
        print(f"Restored best model from memory (val_loss={best_val_loss_monitor:.4f})")

    metrics["best_val"] = best_val_acc if cfg.task == "classification" else best_val_loss
    return model, metrics