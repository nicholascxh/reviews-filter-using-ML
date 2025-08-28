from __future__ import annotations
import pandas as pd

def load_reviews_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic cleaning (extend as needed)
    for col in ["biz_name","biz_cats","biz_desc","review_text"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    if "stars" in df:
        df["stars"] = pd.to_numeric(df["stars"], errors="coerce").fillna(0).astype(int)
    return df
