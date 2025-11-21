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
import copy
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
                # Grab the first whitespace-separated token
                header_token = line[1:].split()[0]

                # FIX: Check for pipes (|) common in UniProt headers
                if "|" in header_token:
                    # e.g. "sp|P04075|ALDOA_HUMAN" -> ["sp", "P04075", "ALDOA_HUMAN"] -> "P04075"
                    current_id = header_token.split("|")[1]
                else:
                    # e.g. "P04075"
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

        # ---------------------------------------------------------
        # DIAGNOSTICS & RESCUE SECTION
        # ---------------------------------------------------------
        seq_dict = read_fasta_to_dict(fasta_path)

        # 1. Diagnostic: Print sample IDs to check formats
        csv_ids = set(agg[id_col])
        fasta_ids = set(seq_dict.keys())

        print("\n--- DEBUGGING ID MISMATCH ---")
        print(f"Total IDs in Table: {len(csv_ids)}")
        print(f"Total IDs in FASTA: {len(fasta_ids)}")

        common = csv_ids.intersection(fasta_ids)
        print(f"Direct Matches: {len(common)}")

        # 2. Attempt Mapping
        agg["seq"] = agg[id_col].map(seq_dict)

        # 3. Rescue Isoforms (e.g., Map 'P04075-2' -> 'P04075')
        missing_mask = agg["seq"].isna()
        n_missing = missing_mask.sum()

        if n_missing > 0:
            print(f"Missing Sequences: {n_missing}")
            # print("Attempting to rescue isoforms (stripping '-2', '-3' suffixes)...")

            # Create a temporary 'clean_id' column (P04075-2 -> P04075)
            clean_ids = agg.loc[missing_mask, id_col].astype(str).apply(lambda x: x.split('-')[0])

            # Map again using the clean ID
            rescued_seqs = clean_ids.map(seq_dict)

            # Fill in the gaps
            agg.loc[missing_mask, "seq"] = rescued_seqs

            # Check how many we saved
            n_still_missing = agg["seq"].isna().sum()
            # print(f"Rescued: {n_missing - n_still_missing}")
            print(f"Still Missing: {n_still_missing}")

            # if n_still_missing > 0:
            # print("Examples of IDs still missing (check FASTA):")
            # print(agg[agg["seq"].isna()][id_col].head(5).tolist())

        # drop rows without sequence
        agg = agg.dropna(subset=["seq"])
        # ---------------------------------------------------------

        out_csv = Path(out_csv)
        agg.to_csv(out_csv, index=False)
        return agg

    # # attach sequences
    # seq_dict = read_fasta_to_dict(fasta_path)
    # agg["seq"] = agg[id_col].map(seq_dict)
    #
    # # drop rows without sequence
    # agg = agg.dropna(subset=["seq"])
    #
    # out_csv = Path(out_csv)
    # agg.to_csv(out_csv, index=False)
    # return agg


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
    lr: float = 1e-3
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
    FIX: Uses padding_idx=1 (Standard ESM) instead of 0 (<cls>).
    """
    toks, labels = zip(*batch)
    lengths = [len(t) for t in toks]
    max_len = max(lengths)

    # FIX: Fill with 1 (<pad>), not 0 (<cls>)
    batch_toks = torch.full(
        (len(toks), max_len), fill_value=1, dtype=torch.long
    )

    for i, t in enumerate(toks):
        batch_toks[i, : len(t)] = t

    labels_tensor = torch.tensor(labels, dtype=torch.long if isinstance(labels[0], int) else torch.float32)
    return batch_toks, labels_tensor


# ---------------------------------------------------------------------
# 4. Model: frozen ESM encoder + MLP head
# ---------------------------------------------------------------------
class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention_weights = nn.Linear(embed_dim, 1)

    def forward(self, x, mask):
        # x: (B, L, D)
        # mask: (B, L)

        # Calculate attention scores
        scores = self.attention_weights(x).squeeze(-1)  # (B, L)

        # Mask padding (set to very low number so softmax ignores them)
        scores = scores.masked_fill(~mask, -1e9)

        # Calculate weights
        attn = torch.softmax(scores, dim=1)  # (B, L)

        # Weighted sum
        # (B, L, 1) * (B, L, D) -> (B, L, D) -> sum -> (B, D)
        weighted = torch.sum(attn.unsqueeze(-1) * x, dim=1)
        return weighted


class NADPHSeqModel(nn.Module):
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

        # 1. Freeze most of the model
        for p in self.esm.parameters():
            p.requires_grad = False

        # 2. SURGICAL UNFREEZING: Allow the last transformer layer to learn
        #    (ESM-2 t33 has 33 layers, indexed 0-32. We unfreeze layer 32)
        for n, p in self.esm.named_parameters():
            if "layers.32" in n:
                p.requires_grad = True

        self.pooler = AttentionPooling(embed_dim)

        # 3. ARCHITECTURE UPGRADE: Add LayerNorm before the head
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
        # NOTE: We now mask against 1, because we padded with 1
        mask = (tokens != 1)

        # Forward pass through ESM
        # (Gradients will flow only through layer 32)
        out = self.esm(tokens, repr_layers=[33], return_contacts=False)
        reps = out["representations"][33]  # (B, L, D)

        # Attention Pooling
        pooled = self.pooler(reps, mask)

        # Normalize
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
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    return model, alphabet


def train_seq_model(
        csv_path: str | Path,
        cfg: NADPHSeqConfig,
) -> Tuple[NADPHSeqModel, Dict[str, float]]:
    """
    Train a sequence-based NADPH responsiveness model.

    Includes:
    - Class weighting for imbalance
    - Early stopping based on validation loss
    - Best model restoration
    """
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
    NUM_WORKERS = 8
    PIN_MEMORY = True if device.type == 'cuda' else False
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn_esm,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn_esm,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )

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

    # 4. Calculate Class Weights (Fixes Imbalance)
    if cfg.task == "classification":
        print("Calculating class weights...")
        # Iterate through the training subset to count labels
        # (Extracting directly from dataset is faster than iterating loader)
        train_labels = [label for _, label in train_ds]
        class_counts = torch.bincount(torch.tensor(train_labels))

        # Avoid division by zero if a class is missing in split
        class_counts = class_counts.float()
        class_counts[class_counts == 0] = 1.0

        # Inverse frequency: weight = 1 / count
        weights = 1.0 / class_counts
        weights = weights / weights.sum()  # Normalize

        print(f"Class weights: {weights}")
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=cfg.lr)

    # 5. Setup Early Stopping
    best_val_acc = -math.inf
    best_val_loss = math.inf  # For tracking regression best
    best_val_loss_monitor = math.inf  # Specifically for early stopping

    patience = 3
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

                loss = criterion(outputs, labels)
                val_loss += loss.item()
                n_val_batches += 1

                if cfg.task == "classification":
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

        val_loss /= max(n_val_batches, 1)

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
        # We stop based on Loss (it's more sensitive than accuracy)
        if val_loss < best_val_loss_monitor:
            best_val_loss_monitor = val_loss
            trigger_times = 0
            # Save the BEST weights to memory
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
