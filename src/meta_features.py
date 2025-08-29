# src/meta_features.py
from __future__ import annotations
import re
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.[\w.-]+", re.I)
PHONE_RE = re.compile(r"\b\+?\d[\d\s().-]{7,}\b")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]", flags=re.UNICODE)

def add_meta_features(df: pd.DataFrame, text_col="review_text") -> pd.DataFrame:
    t = df[text_col].fillna("").astype(str)
    df = df.copy()
    df["text_len"] = t.str.len()
    df["word_count"] = t.str.split().map(len)
    df["avg_word_len"] = (df["text_len"] / df["word_count"].clip(lower=1)).astype(float)
    df["exclaim_count"] = t.str.count(r"!")
    df["question_count"] = t.str.count(r"\?")
    df["uppercase_ratio"] = t.map(lambda s: sum(1 for c in s if c.isupper())) / df["text_len"].clip(lower=1)
    df["url_count"] = t.map(lambda s: len(URL_RE.findall(s)))
    df["phone_count"] = t.map(lambda s: len(PHONE_RE.findall(s)))
    df["email_count"] = t.map(lambda s: len(EMAIL_RE.findall(s)))
    df["emoji_count"] = t.map(lambda s: len(EMOJI_RE.findall(s)))
    return df
