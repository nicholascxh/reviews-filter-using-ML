from __future__ import annotations
import re
import unicodedata
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
    irr_min_anchor_tokens: int
    irr_offtopic_by_cat: Dict[str, List[str]]
    irr_positive_by_cat: Dict[str, List[str]]

    rant_positive_cues: List[str]
    rant_negation_window: int


def load_policy(path: str) -> PolicyConfig:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    irr = y.get("irrelevant", {})
    return PolicyConfig(
        ads_url_regex=y["ads"]["url_regex"],
        ads_phone_regex=y["ads"]["phone_regex"],
        ads_keywords=y["ads"]["keywords"],
        irr_min_cosine=float(irr.get("min_cosine_to_business", 0.25)),
        irr_min_anchor_tokens=int(irr.get("min_anchor_tokens", 1)),
        irr_offtopic_by_cat=irr.get("off_topic_keywords", {}),
        irr_positive_by_cat=irr.get("positive_on_topic", {}),
        rant_positive_cues=y["rant_no_visit"]["positive_cues"],
        rant_negation_window=int(y["rant_no_visit"].get("negation_window", 3)),
    )


_ZW = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_WS = re.compile(r"\s+")

def normalize_text(s: str | None) -> str:
    """Basic normalizer: Unicode NFKC, remove zero-width, collapse whitespace."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _ZW.sub("", s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = _WS.sub(" ", s).strip()
    return s

_WORD = re.compile(r"[a-z0-9']+")

def tokenize(s: str) -> List[str]:
    s = normalize_text(s).lower()
    return _WORD.findall(s)

def _compile_regexes(cfg: PolicyConfig):
    url_re = re.compile("|".join(cfg.ads_url_regex), re.I)
    phone_re = re.compile("|".join(cfg.ads_phone_regex), re.I)
    kw_re = re.compile(r"\b(" + "|".join([re.escape(k) for k in cfg.ads_keywords]) + r")\b", re.I)
    rant_re = re.compile(r"\b(" + "|".join([re.escape(k) for k in cfg.rant_positive_cues]) + r")\b", re.I)
    return url_re, phone_re, kw_re, rant_re


def _cosine_similarity(a: str, b: str) -> float:
    """Lightweight word/biword TF-IDF cosine."""
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, analyzer="word", lowercase=True)
    X = vec.fit_transform([a, b])
    return float(cosine_similarity(X[0], X[1])[0, 0])


def _match_category(biz_cats: str, cat_map: Dict[str, List[str]]) -> List[str]:
    """
    Return the word list (off-topic or positive-on-topic) for the first category key that appears
    inside biz_cats (case-insensitive). If none, return [].
    """
    cats = (biz_cats or "").lower()
    for key, words in cat_map.items():
        if key.lower() in cats:
            return words or []
    return []


def _anchor_overlap(review_text: str, biz_name: str, biz_cats: str, positive_terms: List[str]) -> int:
    """
    Count shared tokens between the review and anchor tokens formed from:
    - business name tokens
    - category tokens
    - positive_on_topic terms (category-specific)
    """
    rv = set(tokenize(review_text))
    anchor_tokens = set(tokenize(biz_name)) | set(tokenize(biz_cats))
    anchor_tokens |= {w.lower() for w in positive_terms}
    # Remove ultra-generic tokens
    anchor_tokens -= {"the", "and", "a", "an", "of", "in", "on", "for", "to"}
    return len(rv & anchor_tokens)


def label_one(
    review_text: str,
    biz_name: str,
    biz_cats: str,
    biz_desc: str,
    cfg: PolicyConfig
) -> Tuple[Dict[str, int], Dict[str, float], List[str]]:
    url_re, phone_re, kw_re, rant_re = _compile_regexes(cfg)

    # Normalize inputs
    t = normalize_text(review_text or "")
    name = normalize_text(biz_name or "")
    cats = normalize_text(biz_cats or "")
    desc = normalize_text(biz_desc or "")

    ads_hit = bool(url_re.search(t) or phone_re.search(t) or kw_re.search(t))

    rant_hit = bool(rant_re.search(t))

    biz_text = " ".join([name, cats, desc]).strip()
    cos = _cosine_similarity(t, biz_text) if (t and biz_text) else 0.0

    off_kw = _match_category(cats, cfg.irr_offtopic_by_cat)  # force irrelevant if present
    pos_terms = _match_category(cats, cfg.irr_positive_by_cat)  # protect from irrelevant if present

    forced_offtopic = False
    if off_kw:
        off_re = re.compile(r"\b(" + "|".join([re.escape(w) for w in off_kw]) + r")\b", re.I)
        if off_re.search(t):
            forced_offtopic = True

    anchor_n = _anchor_overlap(t, name, cats, pos_terms)

    low_cosine = (cos < cfg.irr_min_cosine)
    gated_offtopic = low_cosine and (anchor_n < cfg.irr_min_anchor_tokens)
    irrelevant = bool(forced_offtopic or gated_offtopic)
    relevancy = int(not irrelevant)

    reasons = []
    if ads_hit:
        reasons.append("ads: url/phone/promo")
    if rant_hit:
        reasons.append("rant_no_visit: cue")
    if forced_offtopic:
        reasons.append("irrelevant: category_off_topic")
    elif low_cosine:
        reasons.append(f"irrelevant: cosine<{cfg.irr_min_cosine:.2f} & anchor={anchor_n}")

    labels = {
        "ads": int(ads_hit),
        "irrelevant": int(irrelevant),
        "rant_no_visit": int(rant_hit),
        "relevant": int(relevancy),
    }
    scores = {"cosine_to_business": float(cos), "anchor_tokens": int(anchor_n)}
    return labels, scores, reasons


def predict_batch(df, cfg: PolicyConfig):
    out = []
    for _, row in df.iterrows():
        labels, scores, reasons = label_one(
            review_text=row.get("review_text", ""),
            biz_name=row.get("biz_name", ""),
            biz_cats=row.get("biz_cats", ""),
            biz_desc=row.get("biz_desc", ""),
            cfg=cfg,
        )
        out.append({**labels, **scores, "reasons": "; ".join(reasons)})
    return out
