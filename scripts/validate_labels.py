import pandas as pd
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "data/samples/reviews_sample.csv"
df = pd.read_csv(path)

needed = ["label_relevant","label_irrelevant","label_ads","label_rant_no_visit"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing}")

bad = df[(df["label_relevant"] + df["label_irrelevant"]).isin([0,2])]
# Valid combos are exactly one of them = 1, so sum must be 1

if len(bad):
    print("Found rows with inconsistent relevancy/irrelevant labels:")
    print(bad[["review_id","review_text","label_relevant","label_irrelevant"]])

    df.loc[bad.index, "label_relevant"] = 1 - df.loc[bad.index, "label_irrelevant"]
    out = path.replace(".csv","_fixed.csv")
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\nWrote auto-fixed file to {out}")
else:
    print("All good: relevancy vs. irrelevant labels are consistent.")
