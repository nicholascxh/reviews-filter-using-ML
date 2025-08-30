from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

GT_LABELS = ["label_ads", "label_irrelevant", "label_rant_no_visit"]  # label_relevant is derived

# accept a wide set of pred naming schemes
PRED_NAME_CANDIDATES = [
    ("label_ads", "label_irrelevant", "label_rant_no_visit"),
    ("ads", "irrelevant", "rant_no_visit"),
    ("pred_label_ads", "pred_label_irrelevant", "pred_label_rant_no_visit"),
    ("pred_ads", "pred_irrelevant", "pred_rant_no_visit"),
    ("y_pred_ads", "y_pred_irrelevant", "y_pred_rant_no_visit"),
]

def _ensure_int01(s: pd.Series) -> pd.Series:
    s = s.replace({np.nan: 0})
    s = s.astype(str).str.lower().replace({"true": "1", "false": "0"})
    s = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    return s.clip(0, 1)

def _autolabel_with_rules(gt_df: pd.DataFrame, policy_path: str) -> pd.DataFrame:
    if not policy_path:
        raise SystemExit("Ground truth has no label_* columns. Provide --policy to autolabel with rules.")
    from src.rules_engine import load_policy, predict_batch  # lazy import
    cfg = load_policy(policy_path)
    out = pd.DataFrame(predict_batch(gt_df, cfg))
    gt_df = gt_df.copy()
    gt_df["label_ads"] = out["ads"].astype(int)
    gt_df["label_irrelevant"] = out["irrelevant"].astype(int)
    gt_df["label_rant_no_visit"] = out["rant_no_visit"].astype(int)
    return gt_df

def _pick_gt_cols(df: pd.DataFrame) -> list[str]:
    chosen = []
    for c in GT_LABELS:
        if f"{c}_gt" in df.columns:
            chosen.append(f"{c}_gt")
        elif c in df.columns:
            chosen.append(c)
        else:
            raise SystemExit(f"Ground truth column not found (neither {c} nor {c}_gt).")
    return chosen

def _try_tuple(df: pd.DataFrame, names: tuple[str, str, str]) -> list[str] | None:
    return list(names) if all(n in df.columns for n in names) else None

def _resolve_pred_cols(df: pd.DataFrame, prefer_suffix: str = "") -> list[str]:
    # (1) prefer exact candidate sets with suffix (e.g., *_pred)
    if prefer_suffix:
        for tpl in PRED_NAME_CANDIDATES:
            with_suffix = tuple(f"{n}{prefer_suffix}" for n in tpl)
            got = _try_tuple(df, with_suffix)
            if got: return got
    # (2) then try without suffix
    for tpl in PRED_NAME_CANDIDATES:
        got = _try_tuple(df, tpl)
        if got: return got
    # (3) fallback: suffix-based guessing per label
    guesses = []
    for gt in GT_LABELS:
        suf = gt.split("label_")[-1]
        # prefer *_pred if present
        opts = [c for c in df.columns if c.endswith(suf + prefer_suffix)]
        if not opts:
            opts = [c for c in df.columns if c.endswith(suf)]
        if not opts:
            return []
        guesses.append(opts[0])
    return guesses

def _map_pred_to_gt(df: pd.DataFrame, pred_cols: list[str], gt_name: str, prefer_suffix: str) -> str | None:
    suf = gt_name.split("label_")[-1]  # e.g., 'ads'
    candidates = [
        f"{gt_name}{prefer_suffix}", gt_name,
        f"{suf}{prefer_suffix}", suf,
        f"label_{suf}{prefer_suffix}", f"label_{suf}",
        f"pred_{gt_name}{prefer_suffix}", f"pred_{gt_name}",
        f"pred_{suf}{prefer_suffix}", f"pred_{suf}",
    ]
    for c in candidates:
        if c in pred_cols:
            return c
    for c in candidates:
        if c in df.columns:
            return c
    tail = [c for c in df.columns if c.endswith(suf + prefer_suffix)] or [c for c in df.columns if c.endswith(suf)]
    return tail[0] if tail else None

def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | "":
    for c in candidates:
        if c in df.columns:
            return c
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Ground-truth CSV")
    ap.add_argument("--preds", required=True, help="Predictions CSV")
    ap.add_argument("--policy", default="", help="Policy YAML (used to autolabel GT if missing)")
    ap.add_argument("--dump_errors", default="", help="Optional CSV to dump FP/FN rows")
    args = ap.parse_args()

    gt = pd.read_csv(args.data)
    pr = pd.read_csv(args.preds)

    # If GT labels absent, autolabel using rules (weak labels)
    if any(c not in gt.columns for c in GT_LABELS):
        print("Ground truth label_* columns missing — creating weak labels with Rules...")
        gt = _autolabel_with_rules(gt, args.policy)

    # Align by review_id if present; else align by order
    prefer_suffix = ""
    if "review_id" in gt.columns and "review_id" in pr.columns:
        merged = gt.merge(pr, on="review_id", suffixes=("_gt", "_pred"))
        prefer_suffix = "_pred"
    else:
        merged = gt.copy()
        for c in pr.columns:
            newc = c if c not in merged.columns else f"{c}_pred"
            merged[newc] = pr[c]
        prefer_suffix = "_pred"

    # Build Y (GT) using preferred _gt columns if available
    gt_cols = _pick_gt_cols(merged)
    Y = merged[gt_cols].astype(int).values

    # Resolve prediction columns and map them to GT order
    pred_cols = _resolve_pred_cols(merged, prefer_suffix=prefer_suffix)
    if not pred_cols:
        raise SystemExit("Prediction columns not found. Expected label_* or ads/irrelevant/rant_no_visit style columns.")

    mapped = []
    for gt_name in GT_LABELS:
        col = _map_pred_to_gt(merged, pred_cols, gt_name, prefer_suffix)
        if not col:
            raise SystemExit(f"Could not map prediction column for {gt_name}")
        mapped.append(col)

    Yhat = np.column_stack([_ensure_int01(merged[c]) for c in mapped])

    rep = classification_report(Y, Yhat, target_names=GT_LABELS, zero_division=0, output_dict=True)
    print("\n=== Evaluation ===")
    print(json.dumps(rep, indent=2))

    # Optional FP/FN dump (robust to missing text/meta columns)
    if args.dump_errors:
        # dynamically choose whichever text/meta columns exist
        want_groups = [
            ["review_id"],
            ["business_id", "place_id", "business_name"],  # any
            ["biz_name", "business_name", "name"],
            ["biz_cats", "rating_category", "category", "categories"],
            ["review_text", "text", "comment", "content", "review"],
        ]
        cols_keep = []
        for group in want_groups:
            c = _first_present(merged, group)
            if c:
                cols_keep.append(c)
        # add whichever GT label columns we actually used (gt_cols)
        cols_keep += gt_cols
        cols_keep = [c for c in cols_keep if c in merged.columns]

        # Write errors for the first label with activity
        for i, lab in enumerate(GT_LABELS):
            mask_fp = (Y[:, i] == 0) & (Yhat[:, i] == 1)
            mask_fn = (Y[:, i] == 1) & (Yhat[:, i] == 0)
            if mask_fp.any() or mask_fn.any():
                df_fp = merged.loc[mask_fp, cols_keep].copy()
                df_fp["error_type"] = f"{lab}_FP"
                df_fn = merged.loc[mask_fn, cols_keep].copy()
                df_fn["error_type"] = f"{lab}_FN"
                errs = pd.concat([df_fp, df_fn], axis=0)
                errs.to_csv(args.dump_errors, index=False)
                print(f"Wrote FP/FN sample to {args.dump_errors}")
                break

if __name__ == "__main__":
    main()
