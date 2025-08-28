from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple
import yaml
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
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

def _compile_regexes(cfg: PolicyConfig):
    url_re = re.compile("|".join(cfg.ads_url_regex), re.I)
    phone_re = re.compile("|".join(cfg.ads_phone_regex), re.I)
    kw_re = re.compile(r"\b(" + "|".join([re.escape(k) for k in cfg.ads_keywords]) + r")\b", re.I)
    rant_re = re.compile(r"\b(" + "|".join([re.escape(k) for k in cfg.rant_positive_cues]) + r")\b", re.I)
    return url_re, phone_re, kw_re, rant_re

def _cosine_similarity(a: str, b: str) -> float:
    texts = [a, b]
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, analyzer="word")
    X = vec.fit_transform(texts)
    sim = cosine_similarity(X[0], X[1])[0,0]
    return float(sim)

def label_one(review_text: str, biz_name: str, biz_cats: str, biz_desc: str, cfg: PolicyConfig) -> Tuple[Dict[str,int], Dict[str,float], List[str]]:
    url_re, phone_re, kw_re, rant_re = _compile_regexes(cfg)
    t = review_text or ""

    # ADS
    ads_hit = bool(url_re.search(t) or phone_re.search(t) or kw_re.search(t))

    # RANT NO VISIT (simple positive cue detection)
    rant_hit = bool(rant_re.search(t))

    # IRRELEVANT (low cosine similarity OR obvious off-topic word relative to category)
    biz_text = " ".join([biz_name or "", biz_cats or "", biz_desc or ""]).strip()
    cos = _cosine_similarity(t, biz_text) if (t and biz_text) else 0.0
    irrelevant = cos < cfg.irr_min_cosine

    # Extra off-topic keyword signals (category-aware, light-weight)
    cat = (biz_cats or "").lower()
    off_kw = []
    for k, words in cfg.irr_offtopic_by_cat.items():
        if k in cat:
            off_kw = words
            break
    if off_kw:
        off_re = re.compile(r"\b(" + "|".join([re.escape(w) for w in off_kw]) + r")\b", re.I)
        if off_re.search(t):
            irrelevant = True

    relevancy = int(not irrelevant)

    reasons = []
    if ads_hit: reasons.append("ads: url/phone/promo")
    if rant_hit: reasons.append("rant_no_visit: cue")
    if irrelevant: reasons.append(f"irrelevant: cosine<{cfg.irr_min_cosine:.2f}")

    labels = {"ads": int(ads_hit), "irrelevant": int(irrelevant), "rant_no_visit": int(rant_hit), "relevant": int(relevancy)}
    scores = {"cosine_to_business": cos}
    return labels, scores, reasons

def predict_batch(df, cfg: PolicyConfig):
    out = []
    for _, row in df.iterrows():
        labels, scores, reasons = label_one(
            review_text=row.get("review_text",""),
            biz_name=row.get("biz_name",""),
            biz_cats=row.get("biz_cats",""),
            biz_desc=row.get("biz_desc",""),
            cfg=cfg
        )
        out.append({**labels, **scores, "reasons": "; ".join(reasons)})
    return out
