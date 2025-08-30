# scripts/predict_ensemble.py
from __future__ import annotations
import os, json, argparse, pandas as pd
import numpy as np

from src.rules_engine import load_policy, predict_batch
from src.infer_transformer import TransformerJudge 

def combine_rules_transformer(df: pd.DataFrame, policy_path: str,
                              model_dir: str = "models/transformer/distilbert",
                              thr_path: str = "models/thresholds_transformer.json",
                              mode: str = "precision") -> pd.DataFrame:
    os.makedirs("outputs", exist_ok=True)

    cfg = load_policy(policy_path)
    # Rules pass
    rules_out = predict_batch(df, cfg)  # list of dicts
    rules_df = pd.DataFrame(rules_out)
    rules_df = rules_df.rename(columns={"ads": "ads_rules",
                                        "irrelevant": "irrelevant_rules",
                                        "rant_no_visit": "rant_rules"})
    # Transformer pass (irrelevant only by default; ads/rant probs 0.0 if not trained)
    tj = TransformerJudge(model_dir=model_dir)
    probs = pd.DataFrame(tj.predict_frame(df))
    # Load thresholds
    if os.path.exists(thr_path):
        thr = json.load(open(thr_path, "r"))
    else:
        thr = {"ads": 0.5, "irrelevant": 0.5, "rant_no_visit": 0.5}

    # Decisions
    probs["ads"] = (probs["ads_p"] >= thr.get("ads", 0.5)).astype(int)
    probs["irrelevant"] = (probs["irrelevant_p"] >= thr.get("irrelevant", 0.5)).astype(int)
    probs["rant_no_visit"] = (probs["rant_no_visit_p"] >= thr.get("rant_no_visit", 0.5)).astype(int)

    # Ensemble logic
    out = pd.concat([df.reset_index(drop=True), rules_df, probs], axis=1)

    # ads & rant: rely on rules OR model (model will be 0 for those if not trained)
    out["label_ads"] = (out["ads_rules"].astype(int) | out["ads"].astype(int)).astype(int)
    out["label_rant_no_visit"] = (out["rant_rules"].astype(int) | out["rant_no_visit"].astype(int)).astype(int)

    # irrelevant: precision mode = rules AND model; recall mode = rules OR model
    if mode == "recall":
        out["label_irrelevant"] = (out["irrelevant_rules"].astype(int) | out["irrelevant"].astype(int)).astype(int)
    else:  # precision (default)
        out["label_irrelevant"] = (out["irrelevant_rules"].astype(int) & out["irrelevant"].astype(int)).astype(int)

    # relevant = inverse of irrelevant
    out["label_relevant"] = (1 - out["label_irrelevant"].astype(int))

    # Optional: a compact reason field
    def reason_row(r):
        rs = []
        if r["ads_rules"]: rs.append("ads:rule")
        if r.get("ads", 0): rs.append("ads:model")
        if r["rant_rules"]: rs.append("rant:rule")
        if r.get("rant_no_visit", 0): rs.append("rant:model")
        if r["irrelevant_rules"] and (mode == "precision"):
            rs.append(f"irr:rule+model(p={r['irrelevant_p']:.2f})")
        elif r["irrelevant_rules"] or r.get("irrelevant", 0):
            rs.append(f"irr:rule|model(p={r['irrelevant_p']:.2f})")
        return "; ".join(rs)
    out["reasons"] = out.apply(reason_row, axis=1)

    # Keep only prediction columns + optional scores
    keep = ["review_id","business_id","biz_name","biz_cats","review_text",
            "label_ads","label_irrelevant","label_rant_no_visit","label_relevant",
            "ads_p","irrelevant_p","rant_no_visit_p","reasons"]
    return out[[c for c in keep if c in out.columns]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with canonical columns")
    ap.add_argument("--policy", default="configs/policy.yaml")
    ap.add_argument("--model_dir", default="models/transformer/distilbert")
    ap.add_argument("--thr_path", default="models/thresholds_transformer.json")
    ap.add_argument("--mode", choices=["precision","recall"], default="precision")
    ap.add_argument("--out", default="outputs/predictions_ensemble_transformer.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    for col in ("review_text","biz_name","biz_cats","biz_desc"):
        if col in df.columns:
            df[col] = df[col].replace({np.nan: ""}).astype(str)

    pred = combine_rules_transformer(df, policy_path=args.policy,
                                     model_dir=args.model_dir, thr_path=args.thr_path,
                                     mode=args.mode)
    pred.to_csv(args.out, index=False)
    print(f"Saved ensemble predictions to {args.out}")

if __name__ == "__main__":
    main()
