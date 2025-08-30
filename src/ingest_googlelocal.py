# src/ingest_googlelocal.py
from __future__ import annotations
import argparse, gzip, io, json, os, re, sys, uuid
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

import pandas as pd

# Minimal safe text normalization (avoid importing rules_engine to keep this script standalone)
ZW = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
WS = re.compile(r"\s+")
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = ZW.sub("", s)
    s = s.replace("\r", " ").replace("\n", " ").strip()
    s = WS.sub(" ", s)
    return s

def read_jsonl_gz(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def load_meta(meta_path: str) -> Dict[str, Dict[str, Any]]:
    meta = {}
    for obj in read_jsonl_gz(meta_path):
        gid = obj.get("gmap_id") or obj.get("gmapid") or obj.get("gmapId")
        if not gid:
            continue
        # Try common keys; fall back gracefully
        cats = obj.get("category") or obj.get("categories") or []
        if isinstance(cats, list):
            cats_str = ", ".join([str(c) for c in cats])
        else:
            cats_str = str(cats)
        meta[gid] = {
            "biz_name": obj.get("name") or obj.get("title") or "",
            "biz_cats": cats_str,
            "biz_desc": obj.get("description") or obj.get("desc") or "",
            "state": obj.get("state") or obj.get("province") or "",
            "city": obj.get("city") or "",
            "lat": obj.get("latitude") or obj.get("lat"),
            "lon": obj.get("longitude") or obj.get("lng"),
            "price": obj.get("price") or "",
        }
    return meta

def make_review_id(user_id: str, gmap_id: str, ts: int) -> str:
    base = f"{user_id or 'u'}_{gmap_id or 'g'}_{ts or 't'}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", required=True, help="Path to state reviews JSONL(.gz)")
    ap.add_argument("--meta", required=True, help="Path to state metadata JSONL(.gz)")
    ap.add_argument("--state", default="", help="Optional state code for tagging (e.g., CA)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--max_reviews", type=int, default=0, help="Cap records for testing (0=no cap)")
    args = ap.parse_args()

    os.makedirs(Path(args.out).parent, exist_ok=True)

    meta = load_meta(args.meta)
    rows = []
    n = 0
    for r in read_jsonl_gz(args.reviews):
        gmap_id = r.get("gmap_id") or r.get("gmapid") or r.get("gmapId")
        if not gmap_id:
            continue
        user_id = r.get("user_id") or r.get("uid") or ""
        rating = r.get("rating")
        text = normalize_text(r.get("text", ""))
        time_ms = r.get("time") or r.get("timestamp")  # typically ms epoch
        try:
            ts = int(time_ms)
        except Exception:
            ts = None

        m = meta.get(gmap_id, {})
        row = {
            "review_id": make_review_id(str(user_id), str(gmap_id), ts or 0),
            "business_id": gmap_id,
            "biz_name": normalize_text(m.get("biz_name", "")),
            "biz_cats": normalize_text(m.get("biz_cats", "")),
            "biz_desc": normalize_text(m.get("biz_desc", "")),
            "review_text": text,
            "stars": rating if rating is not None else "",
            "time_ms": ts,
            "state": args.state or m.get("state", ""),
            "city": m.get("city", ""),
            "lat": m.get("lat"),
            "lon": m.get("lon"),
            "price": m.get("price", ""),
        }
        rows.append(row)
        n += 1
        if args.max_reviews and n >= args.max_reviews:
            break

    df = pd.DataFrame(rows)
    # Basic cleanup
    df["review_text"] = df["review_text"].fillna("")
    df["biz_name"] = df["biz_name"].fillna("")
    df["biz_cats"] = df["biz_cats"].fillna("")
    df["biz_desc"] = df["biz_desc"].fillna("")
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")

if __name__ == "__main__":
    main()
