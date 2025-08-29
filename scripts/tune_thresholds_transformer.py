# scripts/tune_thresholds_transformer.py
import argparse, glob, json, numpy as np
from sklearn.metrics import f1_score

def main(args):
    files = sorted(glob.glob(f"{args.model_dir}/cv/fold_outputs_*.npz"))
    if not files:
        raise SystemExit("No fold outputs found. Re-run train_transformer with CV first.")

    y_all, p_all = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        y_all.append(d["ytrue"])
        p_all.append(d["probs"])
    y = np.vstack(y_all)  # (N, C)
    p = np.vstack(p_all)

    best = {}
    for j in range(p.shape[1]):
        ts = np.linspace(0.2, 0.8, 25)
        bf1, bt = -1, 0.5
        for t in ts:
            pred = (p[:, j] >= t).astype(int)
            f1 = f1_score(y[:, j], pred, zero_division=0)
            if f1 > bf1:
                bf1, bt = f1, float(t)
        best[j] = {"thr": bt, "f1": bf1}

    # Map to canonical names (we trained irrelevant-only by default → index 0)
    out = {}
    if p.shape[1] == 1:
        out = {"irrelevant": best[0]["thr"], "ads": 0.5, "rant_no_visit": 0.5}
    else:
        # if multi-label training
        names = np.load(files[0], allow_pickle=True)["labels"].tolist()
        idx = {n:i for i,n in enumerate(names)}
        out = {
            "ads": best[idx.get("label_ads", -1)]["thr"] if "label_ads" in idx else 0.5,
            "irrelevant": best[idx.get("label_irrelevant", -1)]["thr"] if "label_irrelevant" in idx else 0.5,
            "rant_no_visit": best[idx.get("label_rant_no_visit", -1)]["thr"] if "label_rant_no_visit" in idx else 0.5,
        }

    with open("models/thresholds_transformer.json","w") as f:
        json.dump(out, f, indent=2)
    print("Saved transformer thresholds to models/thresholds_transformer.json")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="models/transformer/distilbert")
    args = ap.parse_args()
    main(args)
