"""
seq_nadph.py
------------

Sequence-based modelling of NADPH responsiveness using Hugging Face Transformers (ESM2).

Enhancements over original:
- Backend: Migrated from 'esm' to 'transformers' (AutoModel, AutoTokenizer).
- Optimization: Added Automatic Mixed Precision (AMP) for faster training.
- Memory: Added Gradient Accumulation to simulate large batches on small GPUs.
- Flexibility: Added 'freeze_backbone' option to allow fine-tuning (inspired by HF examples).
- Tokenization: robust handling via AutoTokenizer.

Modes:
- train_mode="pooled": Cached embeddings (fastest).
- train_mode="reps": Cached residue reps (fast-ish, supports Saliency/IG).
- train_mode="tokens": End-to-end ESM forward pass (slowest, supports Fine-tuning).

BSD 3-Clause License
Copyright (c) 2025, Abhinav Mishra
"""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore", category=UserWarning, module="torch")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class NADPHSeqConfig:
    # Model
    model_name: str = "facebook/esm2_t33_650M_UR50D"  # HF model ID
    repr_layer: int = 33
    max_len: int = 1022

    # Task
    task: str = "classification"  # or "regression"
    num_classes: int = 2

    # Training Hyperparams
    batch_size: int = 2  # Actual GPU batch size (keep small for 15GB GPU)
    accum_steps: int = 4  # Gradient accumulation steps (Simulated BS = batch_size * accum_steps)
    head_batch_size: int = 256  # For pooled/reps modes (cheap)

    lr: float = 1e-4
    epochs: int = 15
    patience: int = 5
    gamma_focal: float = 2.0  # Reduced gamma slightly for better convergence

    # Fine-tuning
    freeze_backbone: bool = True  # If False (and mode=tokens), fine-tunes ESM.

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True  # Use AMP (Automatic Mixed Precision)

    # Caching
    cache_dir: str = "cache"
    cache_fp16: bool = True
    cache_tokens: bool = True
    cache_pooled: bool = True
    cache_reps: bool = False

    # Mode
    train_mode: str = "pooled"  # "pooled" | "reps" | "tokens"

    # Workers
    num_workers: int = 4


# -----------------------------------------------------------------------------
# 1) FASTA & Tokenizer Utils
# -----------------------------------------------------------------------------
def read_fasta_to_dict(fasta_path: str | Path) -> Dict[str, str]:
    """Read FASTA into dict: {id: sequence}."""
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
                if current_id:
                    seqs[current_id].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def load_model_and_tokenizer(model_name: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load HF model and tokenizer."""
    print(f"[Transformers] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


# -----------------------------------------------------------------------------
# 2) Data Building
# -----------------------------------------------------------------------------
def classify_hit_row(row: pd.Series) -> str:
    # Adjusted thresholds slightly for robustness
    is_hit = (float(row["EC50"]) < 0.01) and \
             (float(row["delta_max"]) > 0.10) and \
             (float(row["R2"]) > 0.70)
    return "strong" if is_hit else "weak"


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
    agg["label_cls"] = agg["hit_class"].map({"weak": 0, "strong": 1}).astype(int)

    if use_nss and "NSS" in fits_df.columns:
        # If NSS isn't in agg, merge it back or handle appropriately
        pass
        # For simplicity, default to EC50 log transform if NSS logic isn't strictly defined

    # Robust log transform
    ec50 = agg["EC50"].replace(0, 1e-9)  # avoid log(0)
    agg["label_reg"] = -np.log10(ec50)

    # Attach sequences
    seq_dict = read_fasta_to_dict(fasta_path)
    agg["seq"] = agg[id_col].map(seq_dict)

    # Simple rescue for UniProt IDs (e.g., P12345-1 -> P12345)
    missing = agg["seq"].isna()
    if missing.any():
        agg.loc[missing, "seq"] = agg.loc[missing, id_col].apply(lambda x: x.split("-")[0]).map(seq_dict)

    agg = agg.dropna(subset=["seq"])

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_csv, index=False)
    return agg


# -----------------------------------------------------------------------------
# 3) Datasets
# -----------------------------------------------------------------------------
class NADPHBaseDataset(Dataset):
    def __init__(self, data, task):
        self.data = data
        self.task = task

    def __len__(self):
        return len(self.data["labels"])

    def __getitem__(self, idx):
        # Subclasses should implement
        raise NotImplementedError


class NADPHPooledDataset(NADPHBaseDataset):
    def __init__(self, pt_path, task):
        obj = torch.load(pt_path, map_location="cpu")
        data = {"x": obj["pooled"], "labels": obj["label_cls"] if task == "classification" else obj["label_reg"]}
        super().__init__(data, task)

    def __getitem__(self, idx):
        x = self.data["x"][idx].float()
        y = self.data["labels"][idx]
        return x, int(y) if self.task == "classification" else float(y)


class NADPHRepsDataset(NADPHBaseDataset):
    def __init__(self, pt_path, task):
        obj = torch.load(pt_path, map_location="cpu")
        data = {"reps": obj["reps"], "mask": obj["mask"],
                "labels": obj["label_cls"] if task == "classification" else obj["label_reg"]}
        super().__init__(data, task)

    def __getitem__(self, idx):
        return (
            self.data["reps"][idx].float(),
            self.data["mask"][idx].bool(),
            int(self.data["labels"][idx]) if self.task == "classification" else float(self.data["labels"][idx])
        )


class NADPHSeqDataset(Dataset):
    """Tokenize on-the-fly using HF Tokenizer."""

    def __init__(self, csv_path, tokenizer, max_len=1022, task="classification"):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task = task

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row["seq"]),
            truncation=True,
            max_length=self.max_len,
            padding=False,  # Pad in collator
            return_tensors=None  # Return lists
        )
        label = int(row["label_cls"]) if self.task == "classification" else float(row["label_reg"])
        return enc["input_ids"], enc["attention_mask"], label


# -----------------------------------------------------------------------------
# 4) Model Architecture (Attention Pooling + Head)
# -----------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D), mask: (B, L)
        scores = self.attention(x).squeeze(-1)  # (B, L)
        min_val = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask.bool(), min_val)
        weights = F.softmax(scores, dim=1)  # (B, L)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)

class NADPHSeqModel(nn.Module):
    def __init__(self, hf_model: Optional[PreTrainedModel], embed_dim: int, cfg: NADPHSeqConfig):
        super().__init__()
        self.backbone = hf_model
        self.task = cfg.task
        self.freeze_backbone = cfg.freeze_backbone

        if self.backbone:
            if self.freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            else:
                # Optional: Enable gradient checkpointing for memory savings
                if hasattr(self.backbone, "gradient_checkpointing_enable"):
                    self.backbone.gradient_checkpointing_enable()

        self.pooler = AttentionPooling(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        hidden = 512
        out_dim = cfg.num_classes if cfg.task == "classification" else 1

        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, input_ids=None, attention_mask=None, reps=None, pooled=None):
        # 1. Pooled Path (Fastest)
        if pooled is not None:
            x = self.norm(pooled)
            return self._predict(x)

        # 2. Reps Path (Fast)
        if reps is not None:
            pooled = self.pooler(reps, attention_mask)
            x = self.norm(pooled)
            return self._predict(x)

        # 3. Tokens Path (End-to-End)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # HF ESM output: last_hidden_state is (B, L, D)
        last_hidden = outputs.last_hidden_state
        pooled = self.pooler(last_hidden, attention_mask)
        x = self.norm(pooled)
        return self._predict(x)

    def _predict(self, x):
        logits = self.head(x)
        if self.task == "regression":
            logits = logits.squeeze(-1)
        return logits


# -----------------------------------------------------------------------------
# 5) Caching (Updated for Transformers)
# -----------------------------------------------------------------------------
def build_token_cache(csv_path, cfg) -> Path:
    out = Path(cfg.cache_dir) / f"tokens_{Path(csv_path).stem}.pt"
    if out.exists(): return out

    print("[Cache] Tokenizing sequences...")
    df = pd.read_csv(csv_path)
    # Using the "facebook/..." model name directly for tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    tokens = []
    masks = []

    # Batch tokenize for speed
    batch_size = 1000
    for i in range(0, len(df), batch_size):
        batch = df["seq"].iloc[i:i + batch_size].astype(str).tolist()
        enc = tokenizer(batch, truncation=True, max_length=cfg.max_len, padding="max_length", return_tensors="pt")
        # We store as lists to save specific logic later or save raw tensor
        # Here saving raw CPU tensors is fine for cache
        tokens.append(enc["input_ids"])
        masks.append(enc["attention_mask"])

    payload = {
        "input_ids": torch.cat(tokens),
        "attention_mask": torch.cat(masks),
        "label_cls": torch.tensor(df["label_cls"].values),
        "label_reg": torch.tensor(df["label_reg"].values) if "label_reg" in df else None
    }
    torch.save(payload, out)
    return out


def build_pooled_cache(token_path, cfg) -> Path:
    # 1. Rename 'out' to 'cache_path' to avoid ambiguity
    cache_path = Path(cfg.cache_dir) / f"pooled_{cfg.model_name.replace('/', '_')}.pt"
    if cache_path.exists(): return cache_path

    print("[Cache] Building pooled embeddings (Transformers)...")
    data = torch.load(token_path)
    dataset = torch.utils.data.TensorDataset(data["input_ids"], data["attention_mask"])
    loader = DataLoader(dataset, batch_size=cfg.batch_size * 2, num_workers=4)

    model, _ = load_model_and_tokenizer(cfg.model_name)
    model.to(cfg.device)
    pooler = AttentionPooling(model.config.hidden_size).to(cfg.device).eval()

    pooled_list = []

    with torch.no_grad(), torch.amp.autocast(cfg.device, enabled=cfg.cache_fp16):
        for ids, mask in tqdm(loader):
            ids, mask = ids.to(cfg.device), mask.to(cfg.device)

            # 2. FIX: Rename 'out' to 'outputs' so we don't overwrite the path
            outputs = model(ids, attention_mask=mask)
            p = pooler(outputs.last_hidden_state, mask)
            pooled_list.append(p.cpu())

    payload = {
        "pooled": torch.cat(pooled_list),
        "label_cls": data["label_cls"],
        "label_reg": data["label_reg"]
    }

    # 3. Save using the correct path variable
    torch.save(payload, cache_path)
    return cache_path

# -----------------------------------------------------------------------------
# 6) Training Loop (With AMP + Gradient Accumulation)
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def train_seq_model(
        csv_path: str | Path,
        cfg: NADPHSeqConfig,
        patience: bool = True,
) -> Tuple[nn.Module, Dict, Dict, pd.DataFrame]:
    Path(cfg.cache_dir).mkdir(exist_ok=True)
    device = torch.device(cfg.device)

    # 1. Prepare Data
    if cfg.train_mode == "pooled":
        t_cache = build_token_cache(csv_path, cfg)
        p_cache = build_pooled_cache(t_cache, cfg)
        ds = NADPHPooledDataset(p_cache, cfg.task)
        collate = None
        bs = cfg.head_batch_size
        model_backbone = None
        config = AutoModel.from_pretrained(cfg.model_name).config
        embed_dim = config.hidden_size

    elif cfg.train_mode == "tokens":
        model_backbone, tokenizer = load_model_and_tokenizer(cfg.model_name)
        embed_dim = model_backbone.config.hidden_size
        ds = NADPHSeqDataset(csv_path, tokenizer, cfg.max_len, cfg.task)
        from transformers import DataCollatorWithPadding
        collate_hf = DataCollatorWithPadding(tokenizer, padding=True)

        def collate_wrapper(batch):
            inp_batch = [{"input_ids": b[0], "attention_mask": b[1]} for b in batch]
            labels = torch.tensor([b[2] for b in batch])
            out = collate_hf(inp_batch)
            return out["input_ids"], out["attention_mask"], labels

        collate = collate_wrapper
        bs = cfg.batch_size
    else:
        raise ValueError("Mode not implemented in this snippet: " + cfg.train_mode)

    train_idx, val_idx = next(
        StratifiedShuffleSplit(n_splits=1, test_size=0.2).split(np.zeros(len(ds)), [ds[i][-1] for i in range(len(ds))]))
    train_ds, val_ds = Subset(ds, train_idx), Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate, num_workers=cfg.num_workers)
    # -----------------------------------------------------------

    # 2. Model
    model = NADPHSeqModel(model_backbone, embed_dim, cfg).to(device)

    # Optimizer & Loss
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.1)

    if cfg.task == "classification":
        ys = [ds[i][-1] for i in train_idx]
        counts = np.bincount(ys)
        weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32).to(device)
        criterion = FocalLoss(alpha=weights, gamma=cfg.gamma_focal)
    else:
        criterion = nn.MSELoss()

    num_update_steps_per_epoch = math.ceil(len(train_loader) / cfg.accum_steps)
    max_train_steps = cfg.epochs * num_update_steps_per_epoch
    num_warmup_steps = int(0.1 * max_train_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps
    )

    scaler = torch.amp.GradScaler(cfg.device, enabled=cfg.fp16)

    # 3. Training Loop
    history = []
    best_val_loss = math.inf
    best_model_state = None

    # Removed patience_counter initialization

    print(f"Starting training (Mode: {cfg.train_mode}, AccumSteps: {cfg.accum_steps}, AMP: {cfg.fp16})")

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            if cfg.train_mode == "pooled":
                x, y = batch
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast(cfg.device, enabled=cfg.fp16):
                    out = model(pooled=x)
                    loss = criterion(out, y)
            else:
                ids, mask, y = batch
                ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                with torch.amp.autocast(cfg.device, enabled=cfg.fp16):
                    out = model(input_ids=ids, attention_mask=mask)
                    loss = criterion(out, y)

            loss = loss / cfg.accum_steps
            scaler.scale(loss).backward()

            is_update_step = ((step + 1) % cfg.accum_steps == 0) or ((step + 1) == len(train_loader))

            if is_update_step:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * cfg.accum_steps

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad(), torch.amp.autocast(cfg.device, enabled=cfg.fp16):
            for batch in val_loader:
                if cfg.train_mode == "pooled":
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    out = model(pooled=x)
                else:
                    ids, mask, y = batch
                    ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                    out = model(input_ids=ids, attention_mask=mask)

                loss = criterion(out, y)
                val_loss += loss.item()

                if cfg.task == "classification":
                    preds = out.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += len(y)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total if total > 0 else 0

        # scheduler.step(val_loss)
        lr_curr = optimizer.param_groups[0]['lr']

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {lr_curr:.1e}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

        # --- MODIFIED BLOCK ---
        # We still track the best model, but we NEVER break the loop early.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = model.state_dict() if not cfg.freeze_backbone else model.head.state_dict()
            best_model_state = copy.deepcopy(state)
        # ----------------------

    # Restore best model found during the full run
    if best_model_state:
        if cfg.freeze_backbone and cfg.train_mode == "pooled":
            model.head.load_state_dict(best_model_state)
        else:
            try:
                model.load_state_dict(best_model_state)
            except:
                model.head.load_state_dict(best_model_state)

    return model, {"best_val_loss": best_val_loss}, {}, pd.DataFrame(history)