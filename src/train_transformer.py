# src/train_transformer.py
from __future__ import annotations
import os, json, argparse, numpy as np, pandas as pd, torch
from typing import Dict, List
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from torch.nn import BCEWithLogitsLoss
from dataclasses import dataclass

def build_text(df: pd.DataFrame) -> List[str]:
    return (
        df.get("review_text","").fillna("") + " [SEP] " +
        df.get("biz_name","").fillna("") + " | " +
        df.get("biz_cats","").fillna("")
    ).tolist()

@dataclass
class DS:
    encodings: Dict[str, torch.Tensor]
    labels: torch.Tensor
    def __len__(self): return self.labels.shape[0]
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.encodings.items()}
        item["labels"] = self.labels[i]
        return item

class ImbTrainer(Trainer):
    def __init__(self, *args, pos_weight: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device) if self.pos_weight is not None else None)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def make_pos_weight(y: np.ndarray) -> torch.Tensor:
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    w = np.where(pos > 0, (neg / np.clip(pos, 1, None)), 1.0).astype(np.float32)
    return torch.tensor(w)

def run_fold(model_name: str, labels: List[str], df: pd.DataFrame, tr_idx, te_idx, outdir: str,
             max_len: int, epochs: int, lr: float, wd: float, bs: int, seed: int, warmup_ratio: float):
    os.makedirs(os.path.join(outdir, "cv"), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    config = AutoConfig.from_pretrained(model_name, num_labels=len(labels), problem_type="multi_label_classification")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    xtr, xte = df.iloc[tr_idx], df.iloc[te_idx]
    ytr = xtr[labels].astype(int).values
    yte = xte[labels].astype(int).values

    tr_texts = build_text(xtr)
    te_texts = build_text(xte)
    tr_tok = tokenizer(tr_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    te_tok = tokenizer(te_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    ds_tr = DS(tr_tok, torch.tensor(ytr, dtype=torch.float32))

    pos_weight = make_pos_weight(ytr)

    args = TrainingArguments(
        output_dir=os.path.join(outdir, "tmp"),
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=wd,
        logging_steps=50,
        save_strategy="no",
        seed=seed,
        report_to=[],
        dataloader_pin_memory=False,
        warmup_ratio=warmup_ratio,
        max_grad_norm=1.0,             # clip to prevent spikes
        lr_scheduler_type="linear",
    )
    trainer = ImbTrainer(
        model=model,
        args=args,
        train_dataset=ds_tr,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        pos_weight=pos_weight,
    )
    trainer.train()

    # Predict on fold test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.model.to(device)
    with torch.no_grad():
        te_tok_dev = {k: v.to(device) for k, v in te_tok.items()}
        logits = trainer.model(**te_tok_dev).logits.cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= 0.5).astype(int)

    # Save fold outputs for threshold tuning
    np.savez(os.path.join(outdir, "cv", f"fold_outputs_{len(os.listdir(os.path.join(outdir,'cv')))}.npz"),
             ytrue=yte, probs=probs, labels=np.array(labels))

    rep = classification_report(yte, pred, target_names=labels, zero_division=0, output_dict=True)
    return rep

def train_full(model_name: str, labels: List[str], df: pd.DataFrame, outdir: str,
               max_len: int, epochs: int, lr: float, wd: float, bs: int, seed: int, warmup_ratio: float):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    config = AutoConfig.from_pretrained(model_name, num_labels=len(labels), problem_type="multi_label_classification")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    y = df[labels].astype(int).values
    pos_weight = make_pos_weight(y)

    texts = build_text(df)
    tok = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    ds = DS(tok, torch.tensor(y, dtype=torch.float32))

    args = TrainingArguments(
        output_dir=outdir,
        per_device_train_batch_size=bs,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=wd,
        save_strategy="epoch",
        save_total_limit=1,
        seed=seed,
        report_to=[],
        dataloader_pin_memory=False,
        warmup_ratio=warmup_ratio,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
    )
    trainer = ImbTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        pos_weight=pos_weight,
    )
    trainer.train()
    trainer.model.save_pretrained(outdir)
    tokenizer.save_pretrained(outdir)
    with open(os.path.join(outdir, "label_map.json"), "w") as f:
        json.dump({"labels": labels}, f)
    print(f"Saved full model to {outdir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/clean/google_maps_restaurant_reviews_clean.csv")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--outdir", default="models/transformer/distilbert")
    ap.add_argument("--labels", default="label_irrelevant", help="comma-separated label columns to learn")
    ap.add_argument("--max_len", type=int, default=224)   # a bit shorter helps stability
    ap.add_argument("--epochs", type=int, default=3)      # +1 epoch
    ap.add_argument("--lr", type=float, default=3e-5)     # slightly higher
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--train_full", action="store_true")
    args = ap.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    df = pd.read_csv(args.data)
    for c in labels:
        if c not in df.columns:
            raise SystemExit(f"Missing label column: {c}. Run preprocess with --pseudo_label first.")

    gkf = GroupKFold(n_splits=args.cv_folds)
    groups = df["business_id"].astype(str).values

    reps = []
    fold_id = 0
    for i, (tr_idx, te_idx) in enumerate(gkf.split(df, groups=groups), start=1):
        rep = run_fold(
            args.model, labels, df, tr_idx, te_idx, args.outdir,
            args.max_len, args.epochs, args.lr, args.wd, args.bs, args.seed, args.warmup_ratio
        )
        reps.append(rep)
        print(f"\n=== Transformer Fold {i}/{args.cv_folds} ===")
        for k in labels:
            m = rep[k]
            print(f"{k}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1-score']:.3f}")
        print("macro:", {m: round(rep['macro avg'][m], 3) for m in ('precision','recall','f1-score')})
        fold_id += 1

    macro = {m: float(np.mean([r["macro avg"][m] for r in reps])) for m in ("precision","recall","f1-score")}
    print("\n=== Transformer GroupKFold Macro Averages ===")
    print(json.dumps(macro, indent=2))

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "cv_macro.json"), "w") as f:
        json.dump(macro, f, indent=2)

    if args.train_full:
        train_full(args.model, labels, df, args.outdir, args.max_len, args.epochs, args.lr, args.wd, args.bs, args.seed, args.warmup_ratio)

if __name__ == "__main__":
    main()
