import argparse, pandas as pd
from src.data_pipeline import load_reviews_csv
from src.ensemble import combine_rules_ml

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/samples/reviews_sample.csv")
    ap.add_argument("--mode", default="precision")
    ap.add_argument("--out", default="outputs/predictions_ensemble.csv")
    args = ap.parse_args()

    df = load_reviews_csv(args.data)
    out = combine_rules_ml(df, mode=args.mode)
    pd.DataFrame(out).to_csv(args.out, index=False)
    print(f"Saved ensemble predictions to {args.out}")
