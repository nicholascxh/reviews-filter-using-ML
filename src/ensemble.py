# src/ensemble.py
from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd
from src.rules_engine import predict_batch, load_policy
from src.infer_tfidf_lr import TfidfLR
from src.infer_transformer import TransformerJudge

def combine_rules_ml(
    df: pd.DataFrame,
    policy_path: str = "configs/policy.yaml",
    mode: str = "precision",  # "precision" | "recall" | "balanced"
    thresholds: Optional[Dict[str, float]] = None
) -> List[Dict]:
    """
    mode:
      - "precision": ads=OR, rant=OR, irrelevant=AND  (fewer false "irrelevant")
      - "recall":    ads=OR, rant=OR, irrelevant=OR   (catch more off-topic)
      - "balanced":  ads=OR, rant=OR, irrelevant= (ML AND (rules OR high ML prob))
    """
    cfg = load_policy(policy_path)
    rules = predict_batch(df, cfg)           # list of dicts
    ml = TfidfLR()
    ml_out = ml.predict_frame(df)            # list of dicts

    thr = thresholds or {"ads": ml.thr.get("ads", 0.5),
                         "irrelevant": ml.thr.get("irrelevant", 0.5),
                         "rant_no_visit": ml.thr.get("rant_no_visit", 0.5)}

    final = []
    for r, m in zip(rules, ml_out):
        ads_ml = m["ads_p"] >= thr["ads"]
        irr_ml = m["irrelevant_p"] >= thr["irrelevant"]
        rant_ml = m["rant_no_visit_p"] >= thr["rant_no_visit"]

        if mode == "precision":
            irr = int(r["irrelevant"] and irr_ml)
        elif mode == "recall":
            irr = int(r["irrelevant"] or irr_ml)
        else:  # balanced
            irr = int(irr_ml and (r["irrelevant"] or m["irrelevant_p"] >= max(0.6, thr["irrelevant"])))

        ads = int(r["ads"] or ads_ml)
        rant = int(r["rant_no_visit"] or rant_ml)
        rel = int(not irr)

        final.append({
            "ads": ads,
            "irrelevant": irr,
            "rant_no_visit": rant,
            "relevant": rel,
            "cosine_to_business": r.get("cosine_to_business", 0.0),
            "reasons": r.get("reasons","") + f" | ml_p={{ads:{m['ads_p']:.2f}, irr:{m['irrelevant_p']:.2f}, rant:{m['rant_no_visit_p']:.2f}}}"
        })
    return final

def combine_rules_transformer(
    df: pd.DataFrame,
    policy_path: str = "configs/policy.yaml",
    mode: str = "precision",  # "precision" | "balanced" | "recall"
    thresholds: Optional[Dict[str, float]] = None,
    model_dir: str = "models/transformer/distilbert",
) -> List[Dict]:
    cfg = load_policy(policy_path)
    rules = predict_batch(df, cfg)
    tr = TransformerJudge(model_dir=model_dir)
    ml_out = tr.predict_frame(df)

    thr = thresholds or {"ads": 0.5, "irrelevant": 0.5, "rant_no_visit": 0.5}

    final = []
    for r, m in zip(rules, ml_out):
        ads_ml = m["ads_p"] >= thr["ads"]
        irr_ml = m["irrelevant_p"] >= thr["irrelevant"]
        rant_ml = m["rant_no_visit_p"] >= thr["rant_no_visit"]

        if mode == "precision":
            irr = int(r["irrelevant"] and irr_ml)
        elif mode == "recall":
            irr = int(r["irrelevant"] or irr_ml)
        else:
            irr = int(irr_ml and (r["irrelevant"] or m["irrelevant_p"] >= max(0.6, thr["irrelevant"])))

        ads = int(r["ads"] or ads_ml)
        rant = int(r["rant_no_visit"] or rant_ml)
        rel = int(not irr)

        final.append({
            "ads": ads,
            "irrelevant": irr,
            "rant_no_visit": rant,
            "relevant": rel,
            "cosine_to_business": r.get("cosine_to_business", 0.0),
            "reasons": r.get("reasons","") + f" | tr_p={{ads:{m['ads_p']:.2f}, irr:{m['irrelevant_p']:.2f}, rant:{m['rant_no_visit_p']:.2f}}}"
        })
    return final