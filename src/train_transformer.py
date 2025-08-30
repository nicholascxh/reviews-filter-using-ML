from __future__ import annotations
import os, json, argparse
from dataclasses import dataclass
from typing import Dict, List, Iterable

import numpy as np
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss

from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import classification_report

from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)

# ---------------------- Stable dtypes & IO helpers ----------------------

SAFE_DTYPES: Dict[str, str] = {
    "review_text": "string",
    "business_id": "string",
    "biz_name": "string",
    "biz_cats": "string",
    "label_irrelevant": "Int64",
    "label_ads": "Int64",
    "label_rant_no_visit": "Int64",
}

def _empty_series(n: int, dtype: str = "string") -> pd.Series:
    return pd.Series([""] * n, dtype=dtype)

def load_training_frame(
    path: str,
    label_cols: List[str],
    limit: int = 0,
    shuffle_buffer: int = 0,
) -> pd.DataFrame:
    """
    Read only necessary columns, coerce labels -> {0,1}, normalize text.
    - limit: sample N rows (fast dev)
    - shuffle_buffer: random sample without loading whole file (approx via pandas sample if limit < total)
    """
    base_cols = ["review_text", "business_id", "biz_name", "biz_cats"]
    usecols = list(dict.fromkeys(base_cols + label_cols))
    dtypes = {c: SAFE_DTYPES.get(c, "string") for c in usecols}

    df = pd.read_csv(path, usecols=usecols, dtype=dtypes, low_memory=False)

    n = len(df)
    for c in ["review_text", "biz_name", "biz_cats"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype("string")
        else:
            df[c] = _empty_series(n)

    for c in label_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).clip(0, 1).astype(int)

    # keep non-empty
    df = df[df["review_text"].str.len() > 0].reset_index(drop=True)

    # subsample if requested
    if limit and limit > 0 and len(df) > limit:
        if shuffle_buffer and shuffle_buffer > 0:
            frac = min(1.0, max(0.001, limit / float(len(df))))
            df = df.sample(frac=frac, random_state=123).head(limit).reset_index(drop=True)
        else:
            df = df.sample(n=limit, random_state=123).reset_index(drop=True)

    return df

def build_text(df: pd.DataFrame) -> List[str]:
    n = len(df)
    rt = df["review_text"]
    bn = df["biz_name"] if "biz_name" in df.columns else _empty_series(n)
    bc = df["biz_cats"] if "biz_cats" in df.columns else _empty_series(n)
    return (rt + " [SEP] " + bn + " | " + bc).tolist()

# ---------------------- On-the-fly Dataset & Collate ----------------------

@dataclass
class OnTheFlyTextDataset(torch.utils.data.Dataset):
    texts: List[str]
    labels: torch.Tensor  # float32 of shape (N, L)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, i: int):
        # return raw text + label; tokenize in collate_fn
        return {"text": self.texts[i], "labels": self.labels[i]}

class TokenizeCollator:
    """Tokenizes a batch of raw texts on the fly, pads, attaches labels."""
    def __init__(self, tokenizer, max_length: int = 224):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict]):
        texts = [b["text"] for b in batch]
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        enc["labels"] = labels
        return enc

# ---------------------- Trainer with pos_weight BCE ----------------------

class ImbTrainer(Trainer):
    def __init__(self, *args, pos_weight: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def make_pos_weight(y: np.ndarray) -> torch.Tensor:
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    w = np.where(pos > 0, (neg / np.clip(pos, 1, None)), 1.0).astype(np.float32)
    return torch.tensor(w)

# ---------------------- Eval helper ----------------------

@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    batch_size: int = 32,
    max_len: int = 224,
    device: torch.device | None = None,
) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    probs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.detach().cpu().numpy()
        p = 1.0 / (1.0 + np.exp(-logits))
        probs.append(p)
    return np.vstack(probs)

# ---------------------- CV & Full-train flows ----------------------

def run_fold(
    model_name: str, labels: List[str], df: pd.DataFrame,
    tr_idx, te_idx, outdir: str,
    max_len: int, epochs: int, lr: float, wd: float, bs: int, seed: int, warmup_ratio: float,
    grad_accum: int, fp16: bool, bf16: bool, workers: int, fold_id: int
):
    os.makedirs(os.path.join(outdir, "cv"), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    config = AutoConfig.from_pretrained(model_name, num_labels=len(labels), problem_type="multi_label_classification")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    xtr, xte = df.iloc[tr_idx], df.iloc[te_idx]
    ytr = xtr[labels].astype(int).values
    yte = xte[labels].astype(int).values

    tr_texts = build_text(xtr)
    te_texts = build_text(xte)

    ds_tr = OnTheFlyTextDataset(tr_texts, torch.tensor(ytr, dtype=torch.float32))
    collate = TokenizeCollator(tokenizer, max_length=max_len)
    pos_weight = make_pos_weight(ytr)

    args = TrainingArguments(
        output_dir=os.path.join(outdir, "tmp"),
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=bs,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=wd,
        logging_steps=50,
        save_strategy="no",
        seed=seed,
        report_to=[],
        dataloader_pin_memory=False,
        dataloader_num_workers=workers,
        warmup_ratio=warmup_ratio,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        fp16=fp16,
        bf16=bf16,
        remove_unused_columns=False,   
    )

    trainer = ImbTrainer(
        model=model,
        args=args,
        train_dataset=ds_tr,
        eval_dataset=None,
        tokenizer=tokenizer,  # ok on 4.x
        data_collator=collate,
        pos_weight=pos_weight,
    )
    trainer.train()

    probs = predict_probs(trainer.model, tokenizer, te_texts, batch_size=bs, max_len=max_len)
    pred = (probs >= 0.5).astype(int)

    # Save fold outputs
    np.savez(os.path.join(outdir, "cv", f"fold_outputs_{fold_id}.npz"),
             ytrue=yte, probs=probs, labels=np.array(labels))

    # Reporting
    if len(labels) == 1:
        rep_raw = classification_report(yte.ravel(), pred.ravel(), zero_division=0, output_dict=True)
        pos_metrics = rep_raw.get("1", {"precision":0.0,"recall":0.0,"f1-score":0.0,"support":0})
        rep = {
            labels[0]: {
                "precision": float(pos_metrics["precision"]),
                "recall": float(pos_metrics["recall"]),
                "f1-score": float(pos_metrics["f1-score"]),
                "support": int(pos_metrics["support"]),
            },
            "macro avg": {k: float(rep_raw["macro avg"][k]) for k in ("precision","recall","f1-score")}
        }
    else:
        rep = classification_report(yte, pred, target_names=labels, zero_division=0, output_dict=True)

    return rep

def train_full(
    model_name: str, labels: List[str], df: pd.DataFrame, outdir: str,
    max_len: int, epochs: int, lr: float, wd: float, bs: int, seed: int, warmup_ratio: float,
    grad_accum: int, fp16: bool, bf16: bool, workers: int
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    config = AutoConfig.from_pretrained(model_name, num_labels=len(labels), problem_type="multi_label_classification")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    y = df[labels].astype(int).values
    texts = build_text(df)

    ds = OnTheFlyTextDataset(texts, torch.tensor(y, dtype=torch.float32))
    collate = TokenizeCollator(tokenizer, max_length=max_len)
    pos_weight = make_pos_weight(y)

    args = TrainingArguments(
        output_dir=outdir,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=wd,
        save_strategy="epoch",
        save_total_limit=1,
        seed=seed,
        report_to=[],
        dataloader_pin_memory=False,
        dataloader_num_workers=workers,
        warmup_ratio=warmup_ratio,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        fp16=fp16,
        bf16=bf16,
        remove_unused_columns=False,  
    )

    trainer = ImbTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=collate,
        pos_weight=pos_weight,
    )
    trainer.train()
    trainer.model.save_pretrained(outdir)
    tokenizer.save_pretrained(outdir)
    with open(os.path.join(outdir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": labels}, f)
    print(f"Saved full model to {outdir}")

# ---------------------- CLI ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/clean/google_maps_restaurant_reviews_clean.csv")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--outdir", default="models/transformer/distilbert")
    ap.add_argument("--labels", default="label_irrelevant", help="comma-separated label columns to learn")

    ap.add_argument("--max_len", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)

    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--train_full", action="store_true")

    ap.add_argument("--fp16", action="store_true", help="Enable fp16 (CUDA only)")
    ap.add_argument("--bf16", action="store_true", help="Enable bf16 (Ampere+ GPU/CPU)")

    ap.add_argument("--dataloader_workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="Train on N samples (debug/fit)")
    ap.add_argument("--shuffle_buffer", type=int, default=0, help="If >0 and using --limit, randomize sample")

    args = ap.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]

    # Load minimal columns, coerce labels; allow sampling to keep RAM predictable
    df = load_training_frame(args.data, labels, limit=args.limit, shuffle_buffer=args.shuffle_buffer)

    for c in labels:
        if c not in df.columns:
            raise SystemExit(f"Missing label column: {c}. Run pseudo-label/preprocess first.")

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "cv"), exist_ok=True)

    # If neither fp16 nor bf16 explicitly set, choose smart default
    if not args.fp16 and not args.bf16:
        args.fp16 = torch.cuda.is_available()

    # CV (GroupKFold by business if present)
    if not args.train_full:
        if "business_id" in df.columns and df["business_id"].notna().any():
            splitter = GroupKFold(n_splits=max(2, args.cv_folds))
            groups = df["business_id"].astype(str).values
            splits = splitter.split(df, groups=groups)
        else:
            splitter = KFold(n_splits=max(2, args.cv_folds), shuffle=True, random_state=args.seed)
            splits = splitter.split(df)

        reps = []
        for i, (tr_idx, te_idx) in enumerate(splits, start=1):
            rep = run_fold(
                args.model, labels, df, tr_idx, te_idx, args.outdir,
                args.max_len, args.epochs, args.lr, args.wd, args.bs, args.seed,
                args.warmup_ratio, args.grad_accum, args.fp16, args.bf16,
                args.dataloader_workers, fold_id=i
            )
            reps.append(rep)
            print(f"\n=== Transformer Fold {i}/{args.cv_folds} ===")
            for k in labels:
                m = rep[k]
                print(f"{k}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1-score']:.3f}")
            print("macro:", {m: round(rep['macro avg'][m], 3) for m in ('precision','recall','f1-score')})

        macro = {m: float(np.mean([r["macro avg"][m] for r in reps])) for m in ("precision","recall","f1-score")}
        print("\n=== Transformer GroupKFold Macro Averages ===")
        print(json.dumps(macro, indent=2))
        with open(os.path.join(args.outdir, "cv_macro.json"), "w", encoding="utf-8") as f:
            json.dump(macro, f, indent=2)

    # Full train on all data
    if args.train_full:
        train_full(
            args.model, labels, df, args.outdir, args.max_len, args.epochs, args.lr, args.wd,
            args.bs, args.seed, args.warmup_ratio, args.grad_accum, args.fp16, args.bf16,
            args.dataloader_workers
        )

if __name__ == "__main__":
    main()
