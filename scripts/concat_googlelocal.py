from __future__ import annotations
import argparse, glob, pandas as pd, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="data/clean/googlelocal_*.csv")
    ap.add_argument("--out", default="data/clean/googlelocal_multi.csv")
    ap.add_argument("--chunksize", type=int, default=500_000)
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files match: {args.pattern}")

    # write header from the first chunk only
    wrote_header = False
    with open(args.out, "w", encoding="utf-8", newline="") as out:
        for i, f in enumerate(files, 1):
            for j, chunk in enumerate(pd.read_csv(f, chunksize=args.chunksize)):
                chunk.to_csv(out, index=False, header=not wrote_header)
                wrote_header = True
            print(f"[{i}/{len(files)}] appended {os.path.basename(f)}")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
