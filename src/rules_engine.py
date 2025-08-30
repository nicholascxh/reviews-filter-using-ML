from __future__ import annotations
import re, unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple
import yaml
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class PolicyConfig:
    ads_url_regex: List[str]
    ads_phone_regex: List[str]
    ads_keywords: List[str]
    irr_min_cosine: float
    irr_offtopic_by_cat: dict
    rant_positive_cues: List[str]
    rant_negation_window: int

def load_policy(path: str) -> PolicyConfig:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return PolicyConfig(
        ads_url_regex=y["ads"]["url_regex"],
        ads_phone_regex=y["ads"]["phone_regex"],
        ads_keywords=y["ads"]["keywords"],
        irr_min_cosine=float(y["irrelevant"]["min_cosine_to_business"]),
        irr_offtopic_by_cat=y["irrelevant"].get("off_topic_keywords", {}),
        rant_positive_cues=y["rant_no_visit"]["positive_cues"],
        rant_negation_window=int(y["rant_no_visit"]["negation_window"]),
    )

# ---------- text normalization ----------
_ZW = re.compile(r"[\u200B-\u200D\u2060\ufeff]")
_WS = re.compile(r"\s+")
def normalize_text(s) -> str:
    if s is None:
        return ""
    if isinstance(s, float):
        if np.isnan(s):
            return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = _ZW.sub("", s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = _WS.sub(" ", s).strip()
    low = s.lower()
    # map common obfuscations
    low = low.replace("[dot]", ".").replace("(dot)", ".")
    low = low.replace(" d0t ", ".").replace(" dot ", ".")
    return low

# ---------- regex compilers ----------
def _compile_regexes(cfg: PolicyConfig):
    url_re = re.compile("|".join(cfg.ads_url_regex), re.I)
    phone_re = re.compile("|".join(cfg.ads_phone_regex), re.I)
    kw_re = re.compile(r"\b(" + "|".join([re.escape(k) for k in cfg.ads_keywords]) + r")\b", re.I)
    rant_re = re.compile(r"\b(" + "|".join([re.escape(k) for k in cfg.rant_positive_cues]) + r")\b", re.I)
    # precompile off-topic patterns per category key
    off_re_by_cat = {}
    for k, words in (cfg.irr_offtopic_by_cat or {}).items():
        if not words:
            continue
        off_re_by_cat[k.lower()] = re.compile(r"\b(" + "|".join([re.escape(w) for w in words]) + r")\b", re.I)
    return url_re, phone_re, kw_re, rant_re, off_re_by_cat

# ---------- slow cosine (kept for compatibility) ----------
def _cosine_similarity(a: str, b: str) -> float:
    texts = [a, b]
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, analyzer="word")
    X = vec.fit_transform(texts)
    sim = cosine_similarity(X[0], X[1])[0, 0]
    return float(sim)

# ---------- FAST row-wise cosine for a batch ----------
def _rowwise_tfidf_cosine(reviews: List[str], biz_texts: List[str]) -> np.ndarray:
    """
    Fit a single TF-IDF on the concatenated corpus, then compute row-wise cosine
    between review[i] and biz_text[i] efficiently: sum(A*B) since both are L2-normalized.
    """
    n = len(reviews)
    # Combine corpus and fit ONCE
    corpus = reviews + biz_texts
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, analyzer="word")
    X = vec.fit_transform(corpus)  # sparse
    Xr = X[:n]
    Xb = X[n:]
    # L2 normalize rows
    Xr = sk_normalize(Xr, norm="l2", copy=False)
    Xb = sk_normalize(Xb, norm="l2", copy=False)
    # Row-wise cosine: dot product of corresponding rows
    # elementwise sparse multiply then sum per row
    cos = np.array(Xr.multiply(Xb).sum(axis=1)).ravel()
    return cos.astype(float, copy=False)

# ---------- single review (kept for tests / small sets) ----------
def label_one(review_text: str, biz_name: str, biz_cats: str, biz_desc: str, cfg: PolicyConfig) -> Tuple[Dict[str,int], Dict[str,float], List[str]]:
    url_re, phone_re, kw_re, rant_re, off_re_by_cat = _compile_regexes(cfg)
    t = normalize_text(review_text or "")

    # ADS
    ads_hit = bool(url_re.search(t) or phone_re.search(t) or kw_re.search(t))

    # RANT NO VISIT
    rant_hit = bool(rant_re.search(t))

    # IRRELEVANT
    biz_text = normalize_text(" ".join([biz_name or "", biz_cats or "", biz_desc or ""]).strip())
    cos = _cosine_similarity(t, biz_text) if (t and biz_text) else 0.0
    irrelevant = cos < cfg.irr_min_cosine

    # Category-aware off-topic
    cat = (biz_cats or "").lower()
    for k, off_re in off_re_by_cat.items():
        if k in cat and off_re.search(t):
            irrelevant = True
            break

    relevancy = int(not irrelevant)

    reasons = []
    if ads_hit: reasons.append("ads: url/phone/promo")
    if rant_hit: reasons.append("rant_no_visit: cue")
    if irrelevant: reasons.append(f"irrelevant: cosine<{cfg.irr_min_cosine:.2f}")

    labels = {"ads": int(ads_hit), "irrelevant": int(irrelevant), "rant_no_visit": int(rant_hit), "relevant": int(relevancy)}
    scores = {"cosine_to_business": float(cos)}
    return labels, scores, reasons

# ---------- FAST batch labeling (use this for big files) ----------
def predict_batch_fast(df: pd.DataFrame, cfg: PolicyConfig) -> List[Dict]:
    """
    Vectorized policy inference for large batches.
    - Compiles regexes once
    - Computes TF-IDF cosine row-wise in one shot
    Returns list of dicts like predict_batch.
    """
    url_re, phone_re, kw_re, rant_re, off_re_by_cat = _compile_regexes(cfg)

    # Prepare texts
    t = df.get("review_text", pd.Series([""] * len(df))).apply(normalize_text).tolist()
    bname = df.get("biz_name", pd.Series([""] * len(df))).apply(normalize_text).tolist()
    bcats = df.get("biz_cats", pd.Series([""] * len(df))).apply(normalize_text).tolist()
    bdesc = df.get("biz_desc", pd.Series([""] * len(df))).apply(normalize_text).tolist()
    biz = [" ".join([bn or "", bc or "", bd or ""]).strip() for bn, bc, bd in zip(bname, bcats, bdesc)]

    # Cosines in one go
    have_any_biz = any(biz)
    cos = _rowwise_tfidf_cosine(t, biz) if have_any_biz else np.zeros(len(df), dtype=float)

    out: List[Dict] = []
    for i in range(len(df)):
        ti = t[i]
        cat = (bcats[i] or "").lower()

        ads_hit = bool(url_re.search(ti) or phone_re.search(ti) or kw_re.search(ti))
        rant_hit = bool(rant_re.search(ti))
        irrelevant = bool(cos[i] < cfg.irr_min_cosine)

        # off-topic by category
        for k, off_re in off_re_by_cat.items():
            if k in cat and off_re.search(ti):
                irrelevant = True
                break

        relevancy = int(not irrelevant)
        reasons = []
        if ads_hit: reasons.append("ads: url/phone/promo")
        if rant_hit: reasons.append("rant_no_visit: cue")
        if irrelevant: reasons.append(f"irrelevant: cosine<{cfg.irr_min_cosine:.2f}")

        out.append({
            "ads": int(ads_hit),
            "irrelevant": int(irrelevant),
            "rant_no_visit": int(rant_hit),
            "relevant": int(relevancy),
            "cosine_to_business": float(cos[i]),
            "reasons": "; ".join(reasons)
        })
    return out

# ---------- legacy wrapper (still used by tests / Streamlit) ----------
def predict_batch(df: pd.DataFrame, cfg: PolicyConfig) -> List[Dict]:
    # Use the fast path always (it also works for small data)
    return predict_batch_fast(df, cfg)
