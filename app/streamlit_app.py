from __future__ import annotations
import os
import json
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Local imports
from src.rules_engine import load_policy, predict_batch  # rules & policy

# Optional transformer (app works Rules-only if not available)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None


# =========================
# Utilities
# =========================
def coerce_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    s = str(x)
    if s.strip().lower() in {"nan", "none", "null"}:
        return ""
    return s


def normalize_df_text(df: pd.DataFrame, cols=("review_text", "biz_name", "biz_cats", "biz_desc")) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(coerce_str)
        else:
            df[c] = ""
    # Minimal schema guarantees
    if "review_id" not in df.columns:
        df["review_id"] = [f"r{i:08d}" for i in range(len(df))]
    if "business_id" not in df.columns:
        df["business_id"] = ""
    return df


# =========================
# Transformer scoring
# =========================
@st.cache_resource(show_spinner=False)
def load_transformer(model_dir: str):
    if not torch or not AutoTokenizer:
        raise RuntimeError("Transformers not available. Install PyTorch and transformers first.")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tok, model, device


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def transformer_score_irrelevant(
    df: pd.DataFrame,
    model_dir: str,
    text_col: str = "review_text",
    batch_size: int = 32,
    max_length: int = 256
) -> np.ndarray:
    """Return P(irrelevant=1) for each row using a single-label classifier trained for label_irrelevant."""
    tok, model, device = load_transformer(model_dir)
    probs: List[float] = []
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            chunk = df.iloc[i:i + batch_size]
            texts = [coerce_str(t) for t in chunk[text_col].tolist()]
            enc = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            logits = out.logits.detach().cpu().numpy() if hasattr(out, "logits") else out[0].detach().cpu().numpy()
            # If model outputs 2-class logits, take softmax proba for class 1; otherwise sigmoid
            if logits.ndim == 2 and logits.shape[1] == 2:
                p = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
            else:
                p = _sigmoid(logits.squeeze())
            probs.extend(p.tolist())
    return np.array(probs, dtype=float)


# =========================
# Ensemble logic
# =========================
def combine_predictions(
    df: pd.DataFrame,
    rules_df: pd.DataFrame,
    mode: str,
    transformer_probs: np.ndarray | None,
    thr_irrelevant: float = 0.5,
):
    """
    mode options:
      - 'rules'        : ads/rant/irrelevant from Rules only
      - 'transformer'  : only irrelevant from transformer+thr, ads/rant=0
      - 'precision'    : irrelevant = Rules AND (prob>=thr)
      - 'recall'       : irrelevant = Rules OR  (prob>=thr)
    Returns DataFrame columns:
      label_ads, label_irrelevant, label_rant_no_visit, label_relevant, prob_irrelevant, reasons
    """
    ads = rules_df["ads"].astype(int).values
    rant = rules_df["rant_no_visit"].astype(int).values
    irr_rules = rules_df["irrelevant"].astype(int).values
    reasons = rules_df.get("reasons", [""] * len(df))

    if mode == "rules" or transformer_probs is None:
        irr = irr_rules
        prob = np.zeros_like(irr_rules, dtype=float)
    elif mode == "transformer":
        irr_t = (transformer_probs >= thr_irrelevant).astype(int)
        irr = irr_t
        prob = transformer_probs
        ads = np.zeros_like(irr)   # single-label transformer for irrelevant only
        rant = np.zeros_like(irr)
    elif mode == "precision":
        irr_t = (transformer_probs >= thr_irrelevant).astype(int)
        irr = (irr_rules & irr_t).astype(int)
        prob = transformer_probs
    elif mode == "recall":
        irr_t = (transformer_probs >= thr_irrelevant).astype(int)
        irr = (irr_rules | irr_t).astype(int)
        prob = transformer_probs
    else:
        raise ValueError("Unknown mode")

    relevant = (1 - irr).astype(int)
    out = pd.DataFrame({
        "label_ads": ads,
        "label_irrelevant": irr,
        "label_rant_no_visit": rant,
        "label_relevant": relevant,
        "prob_irrelevant": prob,
        "reasons": reasons,
    })
    return out


# =========================
# Metrics helpers
# =========================
def available_gt_columns(df: pd.DataFrame) -> List[str]:
    needed = ["label_ads", "label_irrelevant", "label_rant_no_visit"]
    return [c for c in needed if c in df.columns]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: List[str]) -> Dict:
    from sklearn.metrics import classification_report
    rep = classification_report(y_true, y_pred, target_names=names, zero_division=0, output_dict=True)
    return rep


# =========================
# UI
# =========================
st.set_page_config(page_title="Trustworthy Location Reviews — Demo", layout="wide")

with st.sidebar:
    st.title("Trustworthy Reviews")
    st.caption("ML + Rules to filter ads, rants, and off-topic reviews with explainable reasons.")

    st.header("Settings")
    policy_path = st.text_input(
        "Policy YAML",
        value="configs/policy.yaml",
        help="Rules configuration used by the policy engine."
    )
    model_dir = st.text_input(
        "Transformer model directory",
        value="models/transformer/distilbert",
        help="Folder containing the fine-tuned transformer (e.g., DistilBERT). Leave blank for Rules-only."
    )
    thr_json = st.text_input(
        "Thresholds JSON",
        value="models/thresholds_transformer.json",
        help="Per-label tuned thresholds JSON. If missing, default 0.50 will be used for 'irrelevant'."
    )

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        mode = st.selectbox(
            "Prediction mode",
            ["Ensemble: precision", "Ensemble: recall", "Rules only", "Transformer only"],
            index=0,
            help="Precision = Rules AND Transformer; Recall = Rules OR Transformer."
        )
    with col_m2:
        batch_size = st.slider("Transformer batch size", 8, 64, 32, 8)

    st.markdown("---")
    st.header("Data Controls")
    limit_rows = st.number_input("Row limit for preview (0 = all)", min_value=0, value=2000, step=500)
    use_tuned_thr = st.checkbox("Use threshold from JSON (if available)", value=True)
    thr_override = st.slider("Override threshold for 'irrelevant'", 0.0, 1.0, 0.50, 0.01)
    st.caption("Tip: add `label_*` columns to your CSV to see metrics.")

st.title("Trustworthy Location Reviews — Live Demo")

# File input
colL, colR = st.columns([2, 1])
with colL:
    uploaded = st.file_uploader("Upload a CSV", type=["csv"])
with colR:
    use_sample = st.button("Load sample reviews", help="Loads a small sample for a quick preview.")

if uploaded:
    df = pd.read_csv(uploaded)
elif use_sample:
    sample_path = "data/samples/reviews_sample.csv"
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
    else:
        st.warning("Sample file not found. Please upload your CSV.")
        st.stop()
else:
    st.info("Upload a CSV with at least a `review_text` column (optional: biz_name, biz_cats, biz_desc).")
    st.stop()

if limit_rows and limit_rows > 0:
    df = df.head(int(limit_rows))

# Normalize input columns
df = normalize_df_text(df, cols=("review_text", "biz_name", "biz_cats", "biz_desc"))

# Load policy & run rules
try:
    cfg = load_policy(policy_path)
except Exception as e:
    st.error(f"Failed to load policy at {policy_path}: {e}")
    st.stop()

with st.spinner("Applying policy rules..."):
    rules_list = predict_batch(df, cfg)
rules_df = pd.DataFrame(rules_list)

# Load thresholds
thr_irrelevant = 0.50
if use_tuned_thr and os.path.exists(thr_json):
    try:
        thr = json.load(open(thr_json, "r", encoding="utf-8"))
        thr_irrelevant = float(thr.get("irrelevant", thr_irrelevant))
    except Exception as e:
        st.warning(f"Could not read thresholds from {thr_json} ({e}). Using default 0.50.")
else:
    thr_irrelevant = thr_override

# Optional transformer
transformer_probs = None
want_transformer = (mode in {"Transformer only", "Ensemble: precision", "Ensemble: recall"}) and len(model_dir.strip()) > 0
if want_transformer:
    if AutoTokenizer is None:
        st.warning("Transformers not installed. Falling back to Rules-only.")
        want_transformer = False
    else:
        try:
            with st.spinner("Scoring with transformer (irrelevant)..."):
                transformer_probs = transformer_score_irrelevant(
                    df, model_dir=model_dir, text_col="review_text", batch_size=batch_size, max_length=256
                )
        except Exception as e:
            st.warning(f"Transformer scoring failed ({e}). Falling back to Rules-only.")
            transformer_probs = None
            want_transformer = False

# Ensemble
mode_key = {
    "Rules only": "rules",
    "Transformer only": "transformer",
    "Ensemble: precision": "precision",
    "Ensemble: recall": "recall",
}[mode]
out = combine_predictions(
    df=df,
    rules_df=rules_df,
    mode=mode_key,
    transformer_probs=transformer_probs,
    thr_irrelevant=thr_irrelevant,
)

# Join with original for display
pred_df = pd.concat([df[["review_id", "business_id", "biz_name", "biz_cats", "review_text"]], out], axis=1)

# Optional filters (client-side)
st.subheader("Filters")
colf1, colf2, colf3 = st.columns([2, 1, 1])
with colf1:
    q = st.text_input("Search text", help="Matches review text, business name or category.")
with colf2:
    show_flagged_only = st.checkbox("Show only flagged reviews", value=False)
with colf3:
    sort_by_prob = st.checkbox("Sort by probability (irrelevant)", value=False)

view_df = pred_df.copy()
if q.strip():
    ql = q.lower()
    mask = (
        view_df["review_text"].str.lower().str.contains(ql, na=False) |
        view_df["biz_name"].str.lower().str.contains(ql, na=False) |
        view_df["biz_cats"].str.lower().str.contains(ql, na=False)
    )
    view_df = view_df[mask]
if show_flagged_only:
    view_df = view_df[(view_df["label_ads"] == 1) | (view_df["label_irrelevant"] == 1) | (view_df["label_rant_no_visit"] == 1)]
if sort_by_prob and "prob_irrelevant" in view_df.columns:
    view_df = view_df.sort_values("prob_irrelevant", ascending=False)

# Metrics if GT available
gt_cols = available_gt_columns(df)
st.subheader("Metrics")
if gt_cols:
    y_true = df[gt_cols].astype(int).values
    y_pred = pred_df[["label_ads", "label_irrelevant", "label_rant_no_visit"]].astype(int).values
    rep = compute_metrics(y_true, y_pred, names=gt_cols)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ADS F1", f"{rep['label_ads']['f1-score']:.3f}")
    c2.metric("IRRELEVANT F1", f"{rep['label_irrelevant']['f1-score']:.3f}")
    c3.metric("RANT F1", f"{rep['label_rant_no_visit']['f1-score']:.3f}")
    c4.metric("Macro F1", f"{rep['macro avg']['f1-score']:.3f}")
else:
    st.caption("No ground-truth `label_*` columns found in this CSV. Charts below reflect predictions only.")

# Charts
left, right = st.columns(2)
with left:
    st.markdown("Predicted label counts")
    dist = pred_df[["label_ads", "label_irrelevant", "label_rant_no_visit"]].sum().reset_index()
    dist.columns = ["label", "count"]
    st.bar_chart(dist.set_index("label"))
with right:
    if transformer_probs is not None:
        st.markdown("Transformer P(irrelevant) histogram")
        # Compute histogram in Python for consistent bins
        counts, bins = np.histogram(pred_df["prob_irrelevant"], bins=10, range=(0, 1))
        hist_df = pd.DataFrame({"bin": bins[:-1], "count": counts})
        st.bar_chart(hist_df.set_index("bin"))

# Table (styled highlights)
st.subheader("Predictions")
def color_chip(val: int) -> str:
    return "background-color: #FFEEE9" if int(val) == 1 else ""

show_cols = [
    "review_id", "biz_name", "biz_cats",
    "label_ads", "label_irrelevant", "label_rant_no_visit", "label_relevant",
    "prob_irrelevant", "reasons", "review_text"
]
show_cols = [c for c in show_cols if c in view_df.columns]
styled = view_df[show_cols].copy()
for c in ["label_ads", "label_irrelevant", "label_rant_no_visit", "label_relevant"]:
    if c in styled.columns:
        styled[c] = styled[c].astype(int)

st.dataframe(
    styled.style.applymap(color_chip, subset=["label_ads", "label_irrelevant", "label_rant_no_visit"]),
    use_container_width=True,
    height=480
)

# Downloads
st.markdown("---")
csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download predictions CSV",
    data=csv_bytes,
    file_name="predictions_streamlit.csv",
    mime="text/csv",
)

# Footer
gpu_txt = "Yes" if (torch and torch.cuda.is_available()) else "No"
st.caption(f"Policy: {policy_path} • Model: {model_dir or 'N/A'} • Thr(irrelevant)={thr_irrelevant:.2f} • GPU: {gpu_txt}")
