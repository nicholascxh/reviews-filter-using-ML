# src/featurize_metadata.py
from __future__ import annotations
import argparse, os, re
from typing import Dict, Any
import numpy as np
import pandas as pd

# Vectorized regex patterns for .str.count (strings, not compiled Pattern objects)
URL_PAT   = r"(?:https?://\S+|www\.[\w\.-]+)"
PHONE_PAT = r"\b\+?\d[\d\s\-]{7,}\b"
MULTI_EXCL_PAT = r"!{2,}"
CAPS_PAT  = r"[A-Z]"
DIGIT_PAT = r"\d"
WORD_PAT  = r"\b\S+\b"  # approximates word count faster than split

# Columns we expect and their dtypes (string is cheap & safe here)
DTYPES: Dict[str, Any] = {
    "review_id": "string",
    "business_id": "string",
    "biz_name": "string",
    "biz_cats": "string",
    "biz_desc": "string",
    "review_text": "string",
    "stars": "float32",
    "time_ms": "Int64",        # nullable int
    "state": "string",
    "city": "string",
    "lat": "float32",
    "lon": "float32",
    "price": "string",
    "label_ads": "Int8",
    "label_irrelevant": "Int8",
    "label_rant_no_visit": "Int8",
    "cosine_to_business": "float32",
    "reasons": "string",
}

NEEDED = list(DTYPES.keys())

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add vectorized metadata features."""
    out = df.copy()

    s = out["review_text"].fillna("").astype("string")

    nchar = s.str.len().astype("Int32")
    nword = s.str.count(WORD_PAT).astype("Int32")

    urlc  = s.str.count(URL_PAT, flags=re.IGNORECASE).astype("Int16")
    phc   = s.str.count(PHONE_PAT).astype("Int16")
    exc   = s.str.count(r"!").astype("Int16")
    mexc  = s.str.count(MULTI_EXCL_PAT).astype("Int16")
    ques  = s.str.count(r"\?").astype("Int16")
    caps  = s.str.count(CAPS_PAT).astype("Int32")
    digs  = s.str.count(DIGIT_PAT).astype("Int32")

    # Avoid div/0
    nchar_safe = nchar.mask(nchar == 0, other=1)

    caps_ratio  = (caps / nchar_safe).astype("float32")
    digit_ratio = (digs / nchar_safe).astype("float32")

    out["feat_len_chars"]   = nchar
    out["feat_len_words"]   = nword
    out["feat_url_count"]   = urlc
    out["feat_phone_count"] = phc
    out["feat_exclam"]      = exc
    out["feat_multi_exclam"]= mexc
    out["feat_question"]    = ques
    out["feat_caps_ratio"]  = caps_ratio
    out["feat_digit_ratio"] = digit_ratio

    # time-based
    if "time_ms" in out.columns:
        ts = pd.to_datetime(out["time_ms"], unit="ms", errors="coerce")
        out["feat_hour"] = ts.dt.hour.fillna(-1).astype("Int8")
        out["feat_dow"]  = ts.dt.dayofweek.fillna(-1).astype("Int8")
    else:
        out["feat_hour"] = -1
        out["feat_dow"]  = -1

    return out

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all needed columns exist and cast dtypes."""
    for c in NEEDED:
        if c not in df.columns:
            df[c] = pd.Series([pd.NA] * len(df), dtype=DTYPES[c])
    # Cast to stable dtypes (prevents DtypeWarning)
    for c, dt in DTYPES.items():
        if c in df.columns:
            try:
                df[c] = df[c].astype(dt)
            except Exception:
                # last-resort: coerce to string for text-like, float32 for numeric-ish
                df[c] = df[c].astype("string") if "string" in str(dt) else pd.to_numeric(df[c], errors="coerce")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--limit", type=int, default=0, help="For quick tests; 0 = all")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    wrote_header = False
    total = 0
    chunk_idx = 0
    remaining = args.limit if args.limit > 0 else None

    # Read in chunks with stable dtypes (prevents mixed-type warnings)
    reader = pd.read_csv(
        args.data,
        dtype=DTYPES,
        chunksize=args.chunksize,
        low_memory=False,       # disable mixed-type inference
        encoding="utf-8"
    )

    with open(args.out, "w", encoding="utf-8", newline="") as fout:
        for chunk in reader:
            chunk_idx += 1
            if remaining is not None and remaining <= 0:
                break

            if remaining is not None and len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
            remaining = None if remaining is None else max(0, remaining - len(chunk))

            chunk = ensure_columns(chunk)
            feat_chunk = compute_features(chunk)
            feat_chunk.to_csv(fout, index=False, header=not wrote_header)
            wrote_header = True

            total += len(chunk)
            print(f"[featurize] chunk={chunk_idx} wrote={len(chunk):,} total={total:,}", flush=True)

    print(f"✓ Wrote {args.out} with {total:,} rows", flush=True)

if __name__ == "__main__":
    main()
