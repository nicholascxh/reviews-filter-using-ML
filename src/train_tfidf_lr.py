from __future__ import annotations
import argparse, os, json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
import joblib

LABELS = ["label_ads", "label_irrelevant", "label_rant_no_visit"]

def make_text(df: pd.DataFrame) -> pd.Series:
    return (
        df.get("review_text", "").fillna("") + " [SEP] " +
        df.get("biz_name", "").fillna("") + " " +
        df.get("biz_cats", "").fillna("") + " " +
        df.get("biz_desc", "").fillna("")
    )

def main(args):
    df = pd.read_csv(args.data)
    for c in LABELS:
        if c not in df.columns:
            raise SystemExit(f"Missing ground-truth column: {c}")

    df["text"] = make_text(df)
    y = df[LABELS].astype(int)

    try:
        Xtr, Xte, ytr, yte = train_test_split(
            df["text"], y, test_size=0.2, random_state=42,
            stratify=(y.sum(axis=1) > 0)
        )
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(df["text"], y, test_size=0.2, random_state=42)

    vec = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1, max_features=300_000)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1, max_features=200_000, lowercase=True)),
    ])

    pipe = Pipeline([
        ("vec", vec),
        ("clf", OneVsRestClassifier(LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.5, n_jobs=1
        )))
    ])

    pipe.fit(Xtr, ytr)
    ypred = (pipe.predict_proba(Xte) > 0.5).astype(int)

    print("\n=== TF-IDF(char+word) + LR (holdout) ===")
    print(classification_report(yte, ypred, target_names=LABELS, zero_division=0))

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/tfidf_lr.joblib")
    with open("models/thresholds.json", "w") as f:
        json.dump({"ads": 0.5, "irrelevant": 0.5, "rant_no_visit": 0.5}, f)

    print("Saved model to models/tfidf_lr.joblib and thresholds to models/thresholds.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/samples/reviews_sample.csv")
    main(ap.parse_args())
