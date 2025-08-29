# scripts/cv_eval_tfidf_lr.py
from __future__ import annotations
import argparse, json, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from scipy import sparse

from src.meta_features import add_meta_features

LABELS = ["label_ads", "label_irrelevant", "label_rant_no_visit"]

def make_text(df: pd.DataFrame) -> pd.Series:
    return (
        df.get("review_text","").fillna("") + " [SEP] " +
        df.get("biz_name","").fillna("") + " " +
        df.get("biz_cats","").fillna("") + " " +
        df.get("biz_desc","").fillna("")
    )

def build_vectorizer():
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, max_features=300_000)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=2, max_features=200_000, lowercase=True)
    return FeatureUnion([("char", char_vec), ("word", word_vec)])

def build_clf():
    return OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced", C=1.5, n_jobs=1))

def main(args):
    df = pd.read_csv(args.data)
    for c in LABELS + ["label_relevant"]:
        if c not in df.columns:
            raise SystemExit(f"Missing column {c}. Did you run preprocess with --pseudo_label?")

    # meta features
    df = add_meta_features(df)
    meta_cols = ["text_len","word_count","avg_word_len","exclaim_count","question_count",
                 "uppercase_ratio","url_count","phone_count","email_count","emoji_count"]

    df["text"] = make_text(df)

    X_text = df["text"]
    X_meta = df[meta_cols].fillna(0).astype(float).values
    y = df[LABELS].astype(int).values
    groups = df["business_id"].astype(str).values

    gkf = GroupKFold(n_splits=args.folds)
    reports = []
    all_true, all_pred = [], []

    vec = build_vectorizer()
    clf = build_clf()

    for fold, (tr, te) in enumerate(gkf.split(X_text, y, groups)):
        Xtr_text, Xte_text = X_text.iloc[tr], X_text.iloc[te]
        Xtr_meta, Xte_meta = X_meta[tr], X_meta[te]
        ytr, yte = y[tr], y[te]

        V = vec.fit_transform(Xtr_text)
        Vte = vec.transform(Xte_text)

        # scale meta and hstack
        scaler = StandardScaler(with_mean=False)  # for sparse compatibility
        Mtr = scaler.fit_transform(Xtr_meta)
        Mte = scaler.transform(Xte_meta)

        Xtr = sparse.hstack([V, Mtr], format="csr")
        Xte = sparse.hstack([Vte, Mte], format="csr")

        clf.fit(Xtr, ytr)
        yproba = clf.predict_proba(Xte)
        ypred = (yproba > 0.5).astype(int)

        rep = classification_report(yte, ypred, target_names=LABELS, zero_division=0, output_dict=True)
        reports.append(rep)
        all_true.append(yte)
        all_pred.append(ypred)

        print(f"\n=== Fold {fold+1} / {args.folds} ===")
        print(classification_report(yte, ypred, target_names=LABELS, zero_division=0))

    # aggregate macro
    avg = {}
    for k in ["precision","recall","f1-score"]:
        avg[k] = float(np.mean([r["macro avg"][k] for r in reports]))
    print("\n=== GroupKFold Macro Averages (labels only) ===")
    print(json.dumps(avg, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/clean/google_maps_restaurant_reviews_clean.csv")
    ap.add_argument("--folds", type=int, default=5)
    main(ap.parse_args())
