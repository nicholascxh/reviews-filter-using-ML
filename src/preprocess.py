# src/preprocess.py
from __future__ import annotations
import argparse, hashlib, uuid, re, unicodedata
import pandas as pd
from langdetect import detect, LangDetectException

from src.rules_engine import load_policy, predict_batch

WS = re.compile(r"\s+")
CTRL = re.compile(r"[\u200b\u200c\u200d\ufeff]")

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = CTRL.sub("", s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = WS.sub(" ", s).strip()
    return s

def en_only(s: str) -> bool:
    try:
        return detect(s or "") == "en"
    except LangDetectException:
        return True  # keep if too short/uncertain

def hash_id(s: str, n=12) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()[:n]

def map_kaggle_restaurants(df: pd.DataFrame) -> pd.DataFrame:
    # Expecting columns: business_name, author_name, text, photo, rating, rating_category
    out = pd.DataFrame()
    out["review_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    out["business_id"] = df["business_name"].fillna("").map(lambda x: hash_id(str(x)))
    out["biz_name"] = df["business_name"].fillna("").map(normalize_text)
    out["biz_cats"] = "Restaurant"
    out["biz_desc"] = ""
    out["review_text"] = df["text"].fillna("").map(normalize_text)
    out["stars"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0).astype(int).clip(0, 5)
    # keep extras for analysis
    out["author_name"] = df.get("author_name", "")
    out["rating_category"] = df.get("rating_category", "")
    return out

def add_pseudo_labels(df: pd.DataFrame, policy_path="configs/policy.yaml") -> pd.DataFrame:
    cfg = load_policy(policy_path)
    preds = predict_batch(df, cfg)
    p = pd.DataFrame(preds)
    df = pd.concat([df.reset_index(drop=True), p.reset_index(drop=True)], axis=1)
    # rename to golden-style columns for training
    df = df.rename(columns={
        "ads": "label_ads",
        "irrelevant": "label_irrelevant",
        "rant_no_visit": "label_rant_no_visit",
        "relevant": "label_relevant",
    })
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="path to raw Kaggle CSV")
    ap.add_argument("--out", dest="out", required=True, help="path to cleaned CSV")
    ap.add_argument("--source", choices=["kaggle_restaurants"], default="kaggle_restaurants")
    ap.add_argument("--pseudo_label", action="store_true", help="append rule-based labels")
    args = ap.parse_args()

    raw = pd.read_csv(args.inp)
    if args.source == "kaggle_restaurants":
        df = map_kaggle_restaurants(raw)
    else:
        raise SystemExit("Unknown source")

    # basic cleaning + language filter
    df = df[df["review_text"].astype(str).map(en_only)]
    df = df.drop_duplicates(subset=["business_id", "review_text"]).reset_index(drop=True)

    if args.pseudo_label:
        df = add_pseudo_labels(df)

    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"wrote {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
