# scripts/pseudo_label.py
from __future__ import annotations
import argparse, os, sys
import pandas as pd
from src.rules_engine import load_policy, predict_batch_fast

KEEP_COLS = [
    "review_id","business_id","biz_name","biz_cats","biz_desc",
    "review_text","stars","time_ms","state","city","lat","lon","price"
]

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in KEEP_COLS:
        if c not in df.columns:
            df[c] = None
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Input CSV (large)")
    ap.add_argument("--policy", default="configs/policy.yaml")
    ap.add_argument("--out", required=True, help="Output pseudo-labeled CSV")
    ap.add_argument("--chunksize", type=int, default=200_000, help="Rows per chunk")
    args = ap.parse_args()

    cfg = load_policy(args.policy)

    # Prepare output
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    wrote_header = False
    total = 0
    chunk_idx = 0

    for chunk in pd.read_csv(args.data, chunksize=args.chunksize):
        chunk_idx += 1
        df = ensure_columns(chunk)

        # Fast policy inference for this chunk
        pred_list = predict_batch_fast(df, cfg)
        pred = pd.DataFrame(pred_list)

        lab = pd.DataFrame({
            "label_ads": pred["ads"].astype(int),
            "label_irrelevant": pred["irrelevant"].astype(int),
            "label_rant_no_visit": pred["rant_no_visit"].astype(int),
        })

        out_chunk = pd.concat([df[KEEP_COLS], lab, pred[["cosine_to_business","reasons"]]], axis=1)
        out_chunk.to_csv(args.out, mode="a", index=False, header=(not wrote_header), encoding="utf-8")
        wrote_header = True

        total += len(out_chunk)
        print(f"[{chunk_idx}] wrote {len(out_chunk):,} (total {total:,})", flush=True)

    print(f"✓ Wrote pseudo-labeled file to {args.out} with {total:,} rows", flush=True)

if __name__ == "__main__":
    main()
