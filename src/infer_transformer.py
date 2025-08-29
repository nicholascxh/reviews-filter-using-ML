# src/infer_transformer.py
from __future__ import annotations
import json, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CANON = ["label_ads","label_irrelevant","label_rant_no_visit"]

class TransformerJudge:
    def __init__(self, model_dir="models/transformer/distilbert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        try:
            with open(f"{model_dir}/label_map.json","r") as f:
                lm = json.load(f)
                self.labels = lm.get("labels", CANON)
        except FileNotFoundError:
            self.labels = CANON
        # build index lookup
        self.idx = {lab: i for i, lab in enumerate(self.labels)}

    def _build_text(self, row):
        return f"{row.get('review_text','')} [SEP] {row.get('biz_name','')} | {row.get('biz_cats','')}"

    def predict_frame(self, df: pd.DataFrame):
        texts = df.apply(self._build_text, axis=1).tolist()
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits.detach().cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        out = []
        for p in probs:
            def get(lab):
                if lab in self.idx:
                    return float(p[self.idx[lab]])
                return 0.0
            out.append({
                "ads_p": get("label_ads"),
                "irrelevant_p": get("label_irrelevant"),
                "rant_no_visit_p": get("label_rant_no_visit"),
            })
        return out
