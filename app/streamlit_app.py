from __future__ import annotations
import os
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Local imports
from src.rules_engine import load_policy, predict_batch  # rules & policy
# Transformer imports are optional (app can work Rules-only)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None


# ----------------------------
# Small utilities
# ----------------------------
def coerce_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    s = str(x)
    if s.strip().lower() in {"nan", "none", "null"}:
        return ""
    return s

def normalize_df_text(df: pd.DataFrame, cols=("review_text","biz_name","biz_cats","biz_desc")) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(coerce_str)
        else:
            df[c] = ""
    # Optional: enforce minimal schema columns
    if "review_id" not in df.columns:
        df["review_id"] = [f"r{i:08d}" for i in range(len(df))]
    if "business_id" not in df.columns:
        df["business_id"] = ""
    return df


# ----------------------------
# Transformer scoring
# ----------------------------
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

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def transformer_score_irrelevant(df: pd.DataFrame, model_dir: str, text_col: str = "review_text", batch_size: int = 32) -> np.ndarray:
    """Return P(irrelevant=1) for each row using a single-label classifier trained for label_irrelevant."""
    tok, model, device = load_transformer(model_dir)
    probs: List[float] = []
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            chunk = df.iloc[i:i+batch_size]
            texts = [coerce_str(t) for t in chunk[text_col].tolist()]
            enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            if hasattr(out, "logits"):
                logits = out.logits.detach().cpu().numpy()
            else:
                logits = out[0].detach().cpu().numpy()
            # assume binary single-label: shape (B,2) or (B,)
            if logits.ndim == 2 and logits.shape[1] == 2:
                # class 1 prob via softmax
                p = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
            else:
                # one logit; apply sigmoid
                p = sigmoid(logits.squeeze())
            probs.extend(p.tolist())
    return np.array(probs, dtype=float)


# ----------------------------
# Ensemble logic
# ----------------------------
def combine_predictions(
    df: pd.DataFrame,
    rules_df: pd.DataFrame,
    mode: str,
    transformer_probs: np.ndarray | None,
    thr_irrelevant: float = 0.5,
):
    """
    mode: 'rules', 'transformer', 'precision', 'recall'
      - rules:    ads/rant/irrelevant from Rules only
      - transformer: only irrelevant from transformer+thr, ads/rant=0
      - precision: irrelevant = Rules AND (prob>=thr)
      - recall:    irrelevant = Rules OR  (prob>=thr)
    Returns: DataFrame with columns:
      label_ads, label_irrelevant, label_rant_no_visit, label_relevant,
      prob_irrelevant (if transformer provided), reasons (from rules)
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
        ads = np.zeros_like(irr)  # transformer model is only for irrelevant
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


# ----------------------------
# Metrics (if GT labels are present)
# ----------------------------
def available_gt_columns(df: pd.DataFrame) -> List[str]:
    needed = ["label_ads", "label_irrelevant", "label_rant_no_visit"]
    return [c for c in needed if c in df.columns]

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: List[str]) -> Dict:
    from sklearn.metrics import classification_report
    rep = classification_report(y_true, y_pred, target_names=names, zero_division=0, output_dict=True)
    return rep


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Trustworthy Location Reviews — Demo", page_icon="🧭", layout="wide")

with st.sidebar:
    st.title("🧭 Trustworthy Reviews")
    st.caption("ML + Rules to filter ads, rants, and off-topic reviews")

    st.header("⚙️ Settings")

    policy_path = st.text_input(
        "Policy YAML",
        value="configs/policy.yaml",
        help="Rules config file used by the policy engine."
    )

    model_dir = st.text_input(
        "Transformer model directory",
        value="models/transformer/distilbert",
        help="Folder containing the fine-tuned transformer (e.g., DistilBERT/DeBERTa). Leave blank for Rules-only."
    )

    thr_json = st.text_input(
        "Thresholds JSON",
        value="models/thresholds_transformer.json",
        help="Contains tuned thresholds. If missing, we'll default to 0.5 for 'irrelevant'."
    )

    mode = st.selectbox(
        "Prediction mode",
        ["Ensemble (precision)", "Ensemble (recall)", "Rules only", "Transformer only"],
        index=0,
        help="Precision = stricter (Rules AND Transformer). Recall = broader (Rules OR Transformer)."
    )

    batch_size = st.slider("Transformer batch size", 8, 64, 32, 8)

    st.markdown("---")
    st.caption("Tip: include `label_*` columns in your CSV to see metrics.")

st.title("Trustworthy Location Reviews — Live Demo")

# File input
colL, colR = st.columns([2,1])
with colL:
    uploaded = st.file_uploader("Upload a CSV", type=["csv"])
with colR:
    st.write("")
    st.write("")
    use_sample = st.button("Load sample reviews", help="Loads a small internal sample for a quick preview.")

if uploaded:
    df = pd.read_csv(uploaded)
elif use_sample:
    # Minimal sample for demo; replace path if you want a specific file
    sample_path = "data/samples/reviews_sample.csv"
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
    else:
        st.warning("Sample file not found. Please upload your CSV.")
        st.stop()
else:
    st.info("Upload a CSV with at least a `review_text` column (optionally biz_name, biz_cats, biz_desc).")
    st.stop()

# Normalize input columns
df = normalize_df_text(df, cols=("review_text","biz_name","biz_cats","biz_desc"))

# Load policy & run rules
try:
    cfg = load_policy(policy_path)
except Exception as e:
    st.error(f"Failed to load policy at {policy_path}: {e}")
    st.stop()

with st.spinner("Running policy rules..."):
    rules_list = predict_batch(df, cfg)  # list of dicts
rules_df = pd.DataFrame(rules_list)

# Load thresholds
thr_irrelevant = 0.5
if os.path.exists(thr_json):
    try:
        thr = json.load(open(thr_json, "r", encoding="utf-8"))
        thr_irrelevant = float(thr.get("irrelevant", 0.5))
    except Exception as e:
        st.warning(f"Could not read thresholds from {thr_json} ({e}). Using 0.5.")

# Optional: transformer scoring for irrelevant
transformer_probs = None
want_transformer = mode in {"Transformer only", "Ensemble (precision)", "Ensemble (recall)"} and len(model_dir.strip()) > 0
if want_transformer:
    if AutoTokenizer is None:
        st.error("Transformers not installed. Please `pip install torch transformers`.")
        st.stop()
    try:
        with st.spinner("Loading transformer & scoring (irrelevant)…"):
            transformer_probs = transformer_score_irrelevant(df, model_dir=model_dir, text_col="review_text", batch_size=batch_size)
    except Exception as e:
        st.warning(f"Transformer scoring failed ({e}). Falling back to Rules only.")
        transformer_probs = None
        mode = "Rules only"

# Ensemble
mode_key = {
    "Rules only": "rules",
    "Transformer only": "transformer",
    "Ensemble (precision)": "precision",
    "Ensemble (recall)": "recall",
}[mode]

out = combine_predictions(
    df=df,
    rules_df=rules_df,
    mode=mode_key,
    transformer_probs=transformer_probs,
    thr_irrelevant=thr_irrelevant,
)

# Join with original to display context
pred_df = pd.concat([df[["review_id","business_id","biz_name","biz_cats","review_text"]], out], axis=1)

# Metrics if GT available
gt_cols = available_gt_columns(df)
if gt_cols:
    st.subheader("📊 Metrics (Ground Truth present in your CSV)")
    y_true = df[gt_cols].astype(int).values
    y_pred = pred_df[["label_ads","label_irrelevant","label_rant_no_visit"]].astype(int).values
    rep = compute_metrics(y_true, y_pred, names=gt_cols)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ADS F1", f"{rep['label_ads']['f1-score']:.3f}")
    c2.metric("IRRELEVANT F1", f"{rep['label_irrelevant']['f1-score']:.3f}")
    c3.metric("RANT F1", f"{rep['label_rant_no_visit']['f1-score']:.3f}")
    c4.metric("Macro F1", f"{rep['macro avg']['f1-score']:.3f}")
else:
    st.subheader("📊 Metrics")
    st.caption("No ground-truth `label_*` columns found. Showing descriptive charts only.")

# Quick charts
left, right = st.columns(2)
with left:
    st.markdown("**Predicted label distribution**")
    dist = pred_df[["label_ads","label_irrelevant","label_rant_no_visit"]].sum().reset_index()
    dist.columns = ["label","count"]
    st.bar_chart(dist.set_index("label"))
with right:
    if transformer_probs is not None:
        st.markdown("**Transformer prob(irrelevant) histogram**")
        hist = pd.DataFrame({"prob_irrelevant": pred_df["prob_irrelevant"]})
        st.bar_chart(np.histogram(hist["prob_irrelevant"], bins=10, range=(0,1))[0])

# Table (styled)
st.subheader("🔎 Predictions")
def color_chip(val: int) -> str:
    if val == 1: return "background-color: #FFD6E7"  # light red/pink
    return ""
show_cols = ["review_id","biz_name","biz_cats","label_ads","label_irrelevant","label_rant_no_visit","label_relevant","prob_irrelevant","reasons","review_text"]
show_cols = [c for c in show_cols if c in pred_df.columns]
styled = pred_df[show_cols].copy()
for c in ["label_ads","label_irrelevant","label_rant_no_visit","label_relevant"]:
    if c in styled.columns:
        styled[c] = styled[c].astype(int)
st.dataframe(
    styled.style.applymap(color_chip, subset=["label_ads","label_irrelevant","label_rant_no_visit"]),
    use_container_width=True,
    height=420
)

# Download
csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download predictions CSV",
    data=csv_bytes,
    file_name="predictions_streamlit.csv",
    mime="text/csv",
)

# Footer
st.markdown("---")
gpu = "✅" if (torch and torch.cuda.is_available()) else "❌"
st.caption(f"Policy: `{policy_path}` • Model: `{model_dir or 'N/A'}` • Thr(irrelevant)={thr_irrelevant:.2f} • GPU: {gpu}")
