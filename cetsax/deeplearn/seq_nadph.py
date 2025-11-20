"""
seq_nadph.py
------------

Sequence-based modelling of NADPH responsiveness using protein language models.

This module does:

1. Build a supervised training table by merging:
   - per-protein CETSA summary (EC50, delta_max, R2, hit_class, NSS)
   - protein sequences (FASTA, headers matching 'id')

2. Define a PyTorch Dataset that:
   - uses ESM-2 tokenizer to encode sequences
   - yields (sequence_tokens, label) pairs

3. Define a small MLP classifier/regressor on top of frozen ESM embeddings.

You can use this to:
   - predict strong / medium / weak responders from sequence
   - regress EC50 / NSS from sequence
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


# ---------------------------------------------------------------------
# 1. Utility: read FASTA as id -> sequence
# ---------------------------------------------------------------------

def read_fasta_to_dict(fasta_path: str | Path) -> Dict[str, str]:
    """
    Read a FASTA file into a dict: {id: sequence}.

    Assumes headers like:
        >P12345 some description
    and uses the first whitespace-separated token as the id.
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
                header = line[1:].split()[0]
                current_id = header
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
    ec50_medium: float = 0.5,
    delta_strong: float = 0.10,
    delta_medium: float = 0.08,
    r2_strong: float = 0.70,
    r2_medium: float = 0.50,
) -> str:
    """
    Same classification logic as in hit calling: strong / medium / weak.
    """
    ec50 = float(row["EC50"])
    dm = float(row["delta_max"])
    r2 = float(row["R2"])

    if (ec50 < ec50_strong) and (dm > delta_strong) and (r2 > r2_strong):
        return "strong"
    elif (ec50 < ec50_medium) and (dm > delta_medium) and (r2 > r2_medium):
        return "medium"
    else:
        return "weak"


def build_sequence_supervised_table(
    fits_df: pd.DataFrame,
    fasta_path: str | Path,
    out_csv: str | Path,
    id_col: str = "id",
    use_nss: bool = False,
) -> pd.DataFrame:
    """
    Merge per-protein CETSA features with sequences and labels.

    Steps:
        - Aggregate fits per id (median EC50, delta_max, R2)
        - Derive hit_class (strong/medium/weak)
        - Optionally compute a simple NSS proxy (or pass in if you have it)
        - Attach sequences from FASTA
        - Save to CSV: id, seq, EC50, delta_max, R2, hit_class, label_cls, label_reg

    label_cls : int
        0 = weak, 1 = medium, 2 = strong  (or you can binarize later)

    label_reg : float
        By default: -log10(EC50). If use_nss=True and NSS column exists, label_reg = NSS.
    """
    # Aggregate per id (assuming fits_df has one row per id already, or per id/cond)
    agg = (
        fits_df.groupby(id_col)
        .agg(
            EC50=("EC50", "median"),
            delta_max=("delta_max", "median"),
            R2=("R2", "median"),
        )
        .reset_index()
    )

    # classify
    agg["hit_class"] = agg.apply(classify_hit_row, axis=1)

    # class to int
    class_to_int = {"weak": 0, "medium": 1, "strong": 2}
    agg["label_cls"] = agg["hit_class"].map(class_to_int).astype(int)

    # regression label
    if use_nss and "NSS" in agg.columns:
        agg["label_reg"] = agg["NSS"].astype(float)
    else:
        # -log10 EC50 as affinity-like label
        ec50 = agg["EC50"].replace(0, np.nan)
        agg["label_reg"] = -np.log10(ec50)

    # attach sequences
    seq_dict = read_fasta_to_dict(fasta_path)
    agg["seq"] = agg[id_col].map(seq_dict)

    # drop rows without sequence
    agg = agg.dropna(subset=["seq"])

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
    # for classification
    num_classes: int = 3
    # training
    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class NADPHSeqDataset(Dataset):
    """
    Dataset reading a pre-built CSV:
        id, seq, label_cls, label_reg, ...

    and using an ESM alphabet to tokenize sequences.
    """

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

        # truncate if needed
        if len(seq) > self.max_len:
            seq = seq[: self.max_len]

        # batch_converter expects list of (label, seq)
        _, _, toks = self.batch_converter([(seq_id, seq)])
        toks = toks[0]  # remove batch dim

        if self.task == "classification":
            label = int(row["label_cls"])
            return toks, label
        else:
            label = float(row["label_reg"])
            return toks, label


def collate_fn_esm(batch: List[Tuple[torch.Tensor, int | float]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function to pad ESM token sequences.
    """
    toks, labels = zip(*batch)
    lengths = [len(t) for t in toks]
    max_len = max(lengths)

    batch_toks = torch.full(
        (len(toks), max_len), fill_value=0, dtype=torch.long
    )  # 0 = padding token for ESM
    for i, t in enumerate(toks):
        batch_toks[i, : len(t)] = t

    labels_tensor = torch.tensor(labels, dtype=torch.long if isinstance(labels[0], int) else torch.float32)
    return batch_toks, labels_tensor


# ---------------------------------------------------------------------
# 4. Model: frozen ESM encoder + MLP head
# ---------------------------------------------------------------------

class NADPHSeqModel(nn.Module):
    """
    Simple wrapper:
        - frozen ESM-2 encoder
        - mean-pooled sequence embedding
        - small MLP head for classification or regression
    """

    def __init__(
        self,
        esm_model,
        embed_dim: int,
        task: str = "classification",
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.esm = esm_model
        self.task = task

        for p in self.esm.parameters():
            p.requires_grad = False  # freeze encoder

        hidden = 512
        if task == "classification":
            self.head = nn.Sequential(
                nn.Linear(embed_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

    def forward(
        self,
        tokens: torch.Tensor,
        return_reps: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        tokens: (B, L)

        If return_reps=False:
            returns logits (classification) or scalar predictions (regression).
        If return_reps=True:
            returns (logits, reps, mask), where:
                reps: (B, L, D) final-layer representations
                mask: (B, L) bool, True for non-padding positions
        """
        # NOTE: ESM parameters are still frozen (requires_grad = False),
        # but we allow gradients to flow through reps for saliency/IG.
        out = self.esm(tokens, repr_layers=[33], return_contacts=False)
        reps = out["representations"][33]  # (B, L, D)

        # mean-pool over sequence length (excluding padding = 0)
        mask = (tokens != 0)  # (B, L)
        mask_exp = mask.unsqueeze(-1)      # (B, L, 1)
        reps_masked = reps * mask_exp
        lengths = mask_exp.sum(dim=1)      # (B, 1)
        pooled = reps_masked.sum(dim=1) / lengths.clamp(min=1)

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
    """
    Compute per-residue saliency (gradient norm) for a single batch of tokens.

    Parameters
    ----------
    model : NADPHSeqModel
        Trained model (in eval mode recommended).

    tokens : torch.Tensor
        (B, L) token indices (ESM vocabulary, padding = 0).

    target_class : int or None
        For classification:
            - if None, uses the argmax class per sample
            - if int, uses that class index for all samples
        For regression:
            - must be None; saliency is taken w.r.t. scalar prediction.

    Returns
    -------
    saliency : torch.Tensor
        (B, L) tensor of saliency scores (0 for padding positions).
    """
    model.zero_grad()

    # Forward with representations
    logits, reps, mask = model(tokens, return_reps=True)  # reps: (B, L, D)

    # We want gradient of logit w.r.t. reps
    reps.retain_grad()

    if model.task == "classification":
        # logits: (B, C)
        if target_class is None:
            # use predicted class per sample
            pred_cls = logits.argmax(dim=1)  # (B,)
        else:
            pred_cls = torch.full(
                (logits.size(0),),
                fill_value=int(target_class),
                dtype=torch.long,
                device=logits.device,
            )
        # select logit corresponding to chosen class for each sample
        selected = logits.gather(1, pred_cls.unsqueeze(1)).squeeze(1)  # (B,)
        loss = selected.sum()  # sum over batch → scalar
    else:
        # regression: logits is (B,)
        loss = logits.sum()

    loss.backward()

    # reps.grad: (B, L, D)
    grad = reps.grad  # type: ignore
    # L2 norm over embedding dim
    sal = grad.norm(dim=-1)  # (B, L)

    # zero out padding
    sal = sal * mask.float()
    return sal

def compute_residue_integrated_gradients(
    model: NADPHSeqModel,
    tokens: torch.Tensor,
    target_class: int | None = None,
    steps: int = 50,
) -> torch.Tensor:
    """
    Approximate Integrated Gradients for per-residue importance.

    We treat the final-layer ESM representations as the 'input' to the head
    and integrate grads from a zero baseline to the actual reps.

    Parameters
    ----------
    model : NADPHSeqModel
        Trained model.

    tokens : torch.Tensor
        (B, L) token indices.

    target_class : int or None
        Same semantics as in compute_residue_saliency.

    steps : int
        Number of interpolation steps for IG.

    Returns
    -------
    ig_scores : torch.Tensor
        (B, L) integrated gradient scores for each residue (0 for padding).
    """
    device = next(model.parameters()).device
    model.zero_grad()

    # First, get the actual reps and mask once
    logits, reps, mask = model(tokens.to(device), return_reps=True)  # reps: (B, L, D)
    reps_detached = reps.detach()
    mask = mask.to(device)

    baseline = torch.zeros_like(reps_detached)

    # Accumulate gradients along the path
    total_grad = torch.zeros_like(reps_detached)

    for alpha in torch.linspace(0.0, 1.0, steps, device=device):
        model.zero_grad()
        # Interpolated reps
        reps_interp = baseline + alpha * (reps_detached - baseline)
        reps_interp.requires_grad_(True)

        # Mean-pool and run head manually
        mask_exp = mask.unsqueeze(-1)
        reps_masked = reps_interp * mask_exp
        lengths = mask_exp.sum(dim=1)
        pooled = reps_masked.sum(dim=1) / lengths.clamp(min=1)

        logits_interp = model.head(pooled)
        if model.task == "regression":
            selected = logits_interp  # (B,)
        else:
            if target_class is None:
                # choose argmax per sample (based on original logits)
                base_pred = logits.argmax(dim=1)
                selected = logits_interp.gather(1, base_pred.unsqueeze(1)).squeeze(1)
            else:
                cls_idx = int(target_class)
                selected = logits_interp[:, cls_idx]

        loss = selected.sum()
        loss.backward()

        grad = reps_interp.grad  # (B, L, D)
        total_grad += grad

    # Average gradient and multiply by input difference
    avg_grad = total_grad / float(steps)
    ig = (reps_detached - baseline) * avg_grad  # (B, L, D)

    # Aggregate over embedding dim
    ig_scores = ig.norm(dim=-1) * mask.float()  # (B, L)
    return ig_scores

# ---------------------------------------------------------------------
# 5. Training utilities
# ---------------------------------------------------------------------

def load_esm_model_and_alphabet(model_name: str = "esm2_t33_650M_UR50D"):
    """
    Convenience loader for ESM-2 via fair-esm.

    Requires:
        pip install fair-esm
    """
    try:
        import esm  # type: ignore
    except ImportError as e:
        raise ImportError(
            "You need to install 'fair-esm' (pip install fair-esm) to use this function."
        ) from e

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    return model, alphabet


def train_seq_model(
    csv_path: str | Path,
    cfg: NADPHSeqConfig,
) -> Tuple[NADPHSeqModel, Dict[str, float]]:
    """
    Train a sequence-based NADPH responsiveness model.

    For classification:
        - uses label_cls (0,1,2)
        - CrossEntropyLoss

    For regression:
        - uses label_reg
        - MSELoss

    Returns:
        model, metrics dict
    """
    device = torch.device(cfg.device)

    # load esm stuff
    esm_model, alphabet = load_esm_model_and_alphabet(cfg.model_name)
    esm_model.to(device)

    # dataset & loader
    ds = NADPHSeqDataset(csv_path, alphabet, task=cfg.task, max_len=cfg.max_len)

    # simple train/val split
    n = len(ds)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn_esm,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn_esm,
    )

    # model head
    embed_dim = esm_model.embed_dim
    model = NADPHSeqModel(
        esm_model,
        embed_dim=embed_dim,
        task=cfg.task,
        num_classes=cfg.num_classes,
    ).to(device)

    if cfg.task == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=cfg.lr)

    best_val = math.inf if cfg.task == "regression" else -math.inf
    metrics = {}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for toks, labels in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            toks = toks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(toks)

            if cfg.task == "classification":
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_train_batches += 1

        train_loss /= max(n_train_batches, 1)

        # validation
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

                if cfg.task == "classification":
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                else:
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                n_val_batches += 1

        val_loss /= max(n_val_batches, 1)
        if cfg.task == "classification":
            val_acc = correct / max(total, 1)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
            # track best
            if val_acc > best_val:
                best_val = val_acc
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss

    metrics["best_val"] = best_val
    return model, metrics
