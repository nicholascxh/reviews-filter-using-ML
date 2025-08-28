# Filtering the Noise: ML for Trustworthy Location Reviews

Starter workspace for TikTok TechJam 2025 — ML/NLP system to assess review quality & relevancy and enforce policies (ads, irrelevant content, rants without visit).

## Quickstart

```bash
# 1) Create & activate a virtual environment (Windows PowerShell example)
python -m venv .venv
. .venv/Scripts/Activate.ps1  # on macOS/Linux: source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run unit tests
pytest -q

# 4) Try the Streamlit demo
streamlit run app/streamlit_app.py

# 5) (Optional) Start FastAPI for programmatic inference
uvicorn src.api:app --reload --port 8000
# POST JSON to http://127.0.0.1:8000/predict
```

## What’s inside

```
tiktok-techjam-2025-trustworthy-reviews/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ LICENSE
├─ configs/
│  └─ policy.yaml
├─ data/
│  └─ samples/
│     └─ reviews_sample.csv
├─ outputs/               # created at runtime
├─ app/
│  └─ streamlit_app.py
├─ src/
│  ├─ __init__.py
│  ├─ data_pipeline.py
│  ├─ rules_engine.py
│  ├─ features.py
│  ├─ train_baseline.py
│  ├─ evaluate.py
│  └─ api.py
└─ tests/
   └─ test_rules.py
```

## Goals for the 72-hour sprint
- Day 1: Run baseline (rules + TF-IDF similarity), wire up Streamlit, push to GitHub.
- Day 2: Add transformer fine-tune (optional), LLM judge (optional), improve metrics.
- Day 3: Polish README, record demo video, finalize Devpost.

See comments inside files for TODOs to extend this baseline to an ensemble.
