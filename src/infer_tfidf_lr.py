# src/infer_tfidf_lr.py
from __future__ import annotations
import json
import joblib
import pandas as pd

LABELS_SHORT = ["ads", "irrelevant", "rant_no_visit"]
LABELS_FULL = ["label_ads", "label_irrelevant", "label_rant_no_visit"]

class TfidfLR:
    def __init__(self, model_path="models/tfidf_lr.joblib", thr_path="models/thresholds.json"):
        self.pipe = joblib.load(model_path)
        try:
            with open(thr_path, "r") as f:
                self.thr = json.load(f)
        except FileNotFoundError:
            self.thr = {"ads":0.5, "irrelevant":0.5, "rant_no_visit":0.5}

    def _build_text(self, row):
        return f"{row.get('review_text','')} [SEP] {row.get('biz_name','')} {row.get('biz_cats','')} {row.get('biz_desc','')}"

    def predict_frame(self, df: pd.DataFrame):
        texts = df.apply(self._build_text, axis=1)
        proba = self.pipe.predict_proba(texts)  # shape (n, 3)
        out = []
        for p in proba:
            out.append({
                "ads_p": float(p[0]),
                "irrelevant_p": float(p[1]),
                "rant_no_visit_p": float(p[2]),
            })
        return out
