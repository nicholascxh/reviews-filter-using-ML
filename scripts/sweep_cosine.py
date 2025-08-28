import copy, numpy as np, pandas as pd
from sklearn.metrics import classification_report, f1_score
from src.rules_engine import load_policy, predict_batch

DATA = "data/samples/reviews_sample.csv"  # change to your labeled CSV

df = pd.read_csv(DATA)
y_true = df[["label_ads","label_irrelevant","label_rant_no_visit","label_relevant"]].astype(int).values

def run_at(thr):
    cfg = load_policy("configs/policy.yaml")
    cfg = copy.deepcopy(cfg)
    cfg.irr_min_cosine = float(thr)
    preds = predict_batch(df, cfg)
    p = pd.DataFrame(preds)[["ads","irrelevant","rant_no_visit","relevant"]].astype(int).values
    # return macro F1 and per-label F1s
    f_ads  = f1_score(y_true[:,0], p[:,0], zero_division=0)
    f_irr  = f1_score(y_true[:,1], p[:,1], zero_division=0)
    f_rant = f1_score(y_true[:,2], p[:,2], zero_division=0)
    f_rel  = f1_score(y_true[:,3], p[:,3], zero_division=0)
    return (f_ads + f_irr + f_rant + f_rel)/4.0, (f_ads, f_irr, f_rant, f_rel)

best = (-1, None, None)
for thr in np.linspace(0.10, 0.40, 16):
    macro, per = run_at(thr)
    print(f"thr={thr:.2f} macroF1={macro:.3f}  [ads={per[0]:.3f} irr={per[1]:.3f} rant={per[2]:.3f} rel={per[3]:.3f}]")
    if macro > best[0]:
        best = (macro, thr, per)

print("\nBEST:", best)
