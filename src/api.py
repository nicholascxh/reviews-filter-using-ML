from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from src.rules_engine import load_policy, label_one

app = FastAPI(title="Trustworthy Reviews API", version="0.1.0")
_policy = load_policy("configs/policy.yaml")

class Biz(BaseModel):
    name: str = ""
    categories: str = ""
    description: str = ""

class ReviewReq(BaseModel):
    business: Biz
    review_text: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(req: ReviewReq):
    labels, scores, reasons = label_one(
        review_text=req.review_text,
        biz_name=req.business.name,
        biz_cats=req.business.categories,
        biz_desc=req.business.description,
        cfg=_policy
    )
    return {"labels": labels, "scores": scores, "reasons": reasons}
