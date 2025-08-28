from __future__ import annotations
import argparse, json, os
import pandas as pd
from sklearn.metrics import classification_report
from src.data_pipeline import load_reviews_csv
from src.rules_engine import load_policy, predict_batch

def main(args):
    df = load_reviews_csv(args.data)
    cfg = load_policy(args.policy)

    preds = predict_batch(df, cfg)
    pred_df = pd.DataFrame(preds)

    # Save predictions
    os.makedirs("outputs", exist_ok=True)
    out_csv = os.path.join("outputs", "predictions.csv")
    pred_df.to_csv(out_csv, index=False)

    # If ground-truth labels exist, evaluate
    label_cols = ["label_ads","label_irrelevant","label_rant_no_visit","label_relevant"]
    if all(c in df.columns for c in label_cols):
        y_true = df[label_cols].astype(int)
        y_pred = pred_df.rename(columns={
            "ads":"label_ads",
            "irrelevant":"label_irrelevant",
            "rant_no_visit":"label_rant_no_visit",
            "relevant":"label_relevant"
        })[label_cols].astype(int)
        print("\\n=== Classification Report (Rules Baseline) ===")
        print(classification_report(y_true, y_pred, target_names=label_cols, zero_division=0))
    else:
        print("Labels not found; skipped evaluation.")
    print(f"Saved predictions to {out_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/samples/reviews_sample.csv")
    p.add_argument("--policy", default="configs/policy.yaml")
    main(p.parse_args())
