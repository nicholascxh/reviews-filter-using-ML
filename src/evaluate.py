from __future__ import annotations
import argparse, os, pandas as pd
from sklearn.metrics import classification_report

def main(args):
    df = pd.read_csv(args.data)
    pred = pd.read_csv(args.preds)

    label_cols = ["label_ads","label_irrelevant","label_rant_no_visit","label_relevant"]
    pred_cols = ["ads","irrelevant","rant_no_visit","relevant"]

    if not all(c in df.columns for c in label_cols):
        raise SystemExit("Ground-truth labels not present in data.")

    if not all(c in pred.columns for c in pred_cols):
        raise SystemExit("Prediction columns not present in preds.")

    y_true = df[label_cols].astype(int)
    y_pred = pred[pred_cols].astype(int)
    print("=== Evaluation ===")
    print(classification_report(y_true, y_pred, target_names=label_cols, zero_division=0))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/samples/reviews_sample.csv")
    ap.add_argument("--preds", default="outputs/predictions.csv")
    main(ap.parse_args())
