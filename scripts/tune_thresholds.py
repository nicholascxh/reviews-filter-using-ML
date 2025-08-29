import argparse, json, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from src.data_pipeline import load_reviews_csv
from src.rules_engine import load_policy, predict_batch
from src.infer_tfidf_lr import TfidfLR

def build_preds(df, rules, ml_probs, thr, mode="precision"):
    out = []
    for r, m in zip(rules, ml_probs):
        ads_ml = m["ads_p"] >= thr["ads"]
        irr_ml = m["irrelevant_p"] >= thr["irrelevant"]
        rant_ml = m["rant_no_visit_p"] >= thr["rant_no_visit"]

        if mode == "precision":
            irr = int(r["irrelevant"] and irr_ml)
        elif mode == "recall":
            irr = int(r["irrelevant"] or irr_ml)
        else:
            irr = int(irr_ml and (r["irrelevant"] or m["irrelevant_p"] >= max(0.6, thr["irrelevant"])))

        ads = int(r["ads"] or ads_ml)
        rant = int(r["rant_no_visit"] or rant_ml)
        rel = int(not irr)
        out.append({"ads":ads, "irrelevant":irr, "rant_no_visit":rant, "relevant":rel})
    return pd.DataFrame(out)

def main(args):
    df = load_reviews_csv(args.data)
    y = df[["label_ads","label_irrelevant","label_rant_no_visit","label_relevant"]].astype(int)

    cfg = load_policy("configs/policy.yaml")
    rules = predict_batch(df, cfg)
    ml = TfidfLR()
    ml_probs = ml.predict_frame(df)

    # Start from existing thresholds
    thr = {"ads": ml.thr.get("ads",0.5),
           "irrelevant": ml.thr.get("irrelevant",0.5),
           "rant_no_visit": ml.thr.get("rant_no_visit",0.5)}

    search = np.linspace(0.30, 0.80, 11)  # 0.30, 0.35, ..., 0.80
    best = {}

    # Tune each label independently (keeps logic simple)
    for label, key in [("label_ads","ads"), ("label_irrelevant","irrelevant"), ("label_rant_no_visit","rant_no_visit")]:
        best_f1, best_t = -1.0, thr[key]
        for t in search:
            tmp = dict(thr); tmp[key] = float(t)
            pred = build_preds(df, rules, ml_probs, tmp, mode="precision")
            f1 = f1_score(y[label], pred[key], zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        best[key] = {"thr": best_t, "f1": best_f1}

    # Save thresholds.json
    out = {k: v["thr"] for k,v in best.items()}
    with open("models/thresholds.json","w") as f:
        json.dump(out, f, indent=2)

    print("Best thresholds:", json.dumps(best, indent=2))
    print("Saved to models/thresholds.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/samples/reviews_sample.csv")
    main(ap.parse_args())
