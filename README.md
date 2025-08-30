# PetaByteSized — Trustworthy Location Reviews

**Noise filtering for local reviews.** Detect **ads**, **irrelevant content**, and **rants from non‑visitors** with a YAML‑driven policy engine, classical ML baselines, and a transformer fine‑tuned for relevancy — all wrapped in a reproducible pipeline and Streamlit app.

---

## Elevator pitch

Keep location reviews **trustworthy** by automatically **flagging content that breaks policy** or is **off-topic** for the place. PetaByteSized combines transparent **rules** (with human‑readable reasons) and **ML** (TF‑IDF + a transformer) into a precision‑first **ensemble** suitable for moderation workflows.

---

## About the project

### Problem we address
Online reviews shape where people eat, shop, and visit — but **promotions**, **off‑topic rants**, and **reviews from people who never visited** distort reality. We built an ML system that:
- **Gauges quality**: flags spam/ads, promotion links, phone “call now” content.
- **Assesses relevancy**: checks if a review genuinely relates to the business.
- **Enforces policy**: detects rant‑without‑visit and other violations, returning clear reasons.

### What inspired us
We wanted a pragmatic pipeline that a real platform could deploy quickly: **explicit policies** for accountability + **ML** for generalization. We optimized for **explainability**, **reproducibility**, and **speed to value** in a 72‑hour window.

### What we built
- **Policy rules engine** (`configs/policy.yaml`, `src/rules_engine.py`)
  - Ads: URL/phone/keyword regexes.
  - Irrelevant: semantic mismatch vs business text + category‑aware off‑topic lists.
  - Rant‑no‑visit: cue phrases with a small negation window.
  - Outputs **human‑readable reasons** (e.g., `ads: url/phone/promo`, `irrelevant: cosine<thr`, `rant_no_visit: cue`).
- **Baselines**
  - Rules‑only baseline.
  - TF‑IDF + Logistic Regression (per‑label thresholds).
- **Transformer**
  - DistilBERT fine‑tuned **for `label_irrelevant` only** (class imbalance handled with pos‑weighted BCE).
- **Ensemble**
  - Precision‑oriented: **(rules OR transformer_above_threshold)**.
- **Interactive app**
  - Streamlit UI to upload CSVs, run predictions, view reasons, and export results.
- **Reproducible CLIs**
  - Ingest, clean, pseudo‑label, train, predict, evaluate, and tune thresholds.

### Challenges & constraints we acknowledge
- **Only “irrelevant” is model‑trained.**  
  Given time/compute limits, we **focused the transformer on relevancy**. `label_ads` and `label_rant_no_visit` are rules‑only in this version.
- **Weak labels via pseudo‑labeling.**  
  For large datasets we generated labels using our rules to **scale quickly**. This improves iteration speed but **introduces label bias**. Evaluations on these sets can **over‑estimate** performance for rule‑like patterns.
- **Limited human gold.**  
  We used a small, hand‑curated sample for sanity checks. A larger human‑labeled set would further validate generalization.
- **Compute/logistics.**  
  We optimized training for Windows/CPU and modest GPUs and added knobs for dataset limiting, workers, and sequence length.

### What we learned
We went through the end‑to‑end ML lifecycle under hackathon constraints:
1. **Data**: acquire (Kaggle + McAuley Google Local), clean, standardize schemas.
2. **Labeling**: bootstrap with **pseudo‑labels** when gold is scarce.
3. **Modeling**: start with explainable rules → add **TF‑IDF** → fine‑tune a **transformer** on the hardest task (`irrelevant`).
4. **Thresholds/Ensembles**: pick operating points for precision/recall and compose with rules.
5. **Evaluation**: report metrics, inspect error dumps, and **state limitations** clearly.

---

## Datasets, tools & tech

### Assets & datasets
- **Kaggle**: Google Maps Restaurant Reviews (text, 1.1k items).  
- **McAuley Google Local**: Large‑scale Google Local review dumps (we ingested CA, NY, TX subsets).
- **Small curated sample**: for unit tests and sanity metrics.

### Development tools
- VS Code, Python venv, Streamlit, GitHub for revision control.

### Libraries & frameworks
- **Hugging Face Transformers**, **PyTorch**
- **scikit‑learn**, **pandas**, **numpy**
- **Streamlit** for the UI
- **PyYAML** + **regex** for policy rules

### APIs (optional/replaceable)
- OpenAI/GPT family for auxiliary pseudo‑label sanity checks or error analysis (not required for core runs).

---

## Repository structure

```
configs/
  policy.yaml                     # Policy rules (ads / irrelevant / rant)

data/
  raw/                            # raw downloads
  clean/                          # cleaned + pseudo-labeled CSVs
  samples/                        # small curated sample for unit tests & demo

scripts/
  ingest_googlelocal.py           # (optional) downloader/merger for McAuley Google Local
  pseudo_label.py                 # rules → weak labels
  tune_thresholds_transformer.py  # sweeps thresholds from CV fold outputs
  predict_ensemble.py             # combine rules + transformer predictions
  concat_googlelocal.py
  cv_eval_tfidf_lr.py
  sweep_cosine.py
  tune_threshold.py
  validate_labels.py

src/
  rules_engine.py                 # YAML-configured policy engine
  data_pipeline.py                # cleaning & standardization
  featurize_metadata.py           # adds text+metadata features
  train_baseline.py               # rules baseline runner
  train_tfidf_lr.py               # TF-IDF + Logistic Regression
  train_transformer.py            # DistilBERT trainer (CV/full; Windows-friendly)
  evaluate.py                     # metrics + error slice dumps
  api.py
  ensemble.py
  infer_tfidf_lr.py
  infer_transformer.py
  ingest_googlelocal.py
  preprocess.py

app/
  streamlit_app.py                # interactive UI

tests/
  test_rules.py                   # unit tests for policy rules
```

---

## Setup

**Windows PowerShell**

```powershell
python -m venv .venv
Set-ExecutionPolicy Bypass -Scope Process -Force
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path
pip install -r requirements.txt
```

> If activation was previously blocked, the `Bypass` line fixes it for this session.

---

## Quick start (small sample)

**Run rules baseline**

```bash
python src/train_baseline.py --data data/samples/reviews_sample.csv
```

**TF‑IDF + Logistic Regression**

```bash
python src/train_tfidf_lr.py --data data/samples/reviews_sample.csv
```

**Ensemble predict + evaluate**

```bash
python scripts/predict_ensemble.py \
  --data data/samples/reviews_sample.csv \
  --policy configs/policy.yaml \
  --model_dir models/transformer/distilbert \
  --thr_path models/thresholds_transformer.json \
  --mode precision \
  --out outputs/predictions.csv

python src/evaluate.py \
  --data data/samples/reviews_sample.csv \
  --preds outputs/predictions.csv \
  --policy configs/policy.yaml \
  --dump_errors outputs/errors_sample.csv
```

**Streamlit app**

```bash
streamlit run app/streamlit_app.py
```

Expected CSV columns for upload:

```
review_id,business_id,biz_name,biz_cats,biz_desc,review_text,stars
```

---

## Large‑scale pipeline (Google Local CA/NY/TX)

**Ingest & merge**

```bash
python scripts/ingest_googlelocal.py --states TX NY CA --out data/clean/

# Merge all ingested shards (PowerShell-friendly: run this as a script)
python - <<'PY'
import pandas as pd, glob
files = glob.glob("data/clean/googlelocal_*.csv")
pd.concat([pd.read_csv(f) for f in files], ignore_index=True)\
  .to_csv("data/clean/googlelocal_multi.csv", index=False)
print("Wrote data/clean/googlelocal_multi.csv")
PY
```

**Pseudo‑label (weak labels from rules)**

```bash
python scripts/pseudo_label.py \
  --data data/clean/googlelocal_multi.csv \
  --policy configs/policy.yaml \
  --out data/clean/googlelocal_multi_labeled.csv
```

**(Optional) Feature engineering**

```bash
python src/featurize_metadata.py \
  --data data/clean/googlelocal_multi_labeled.csv \
  --out data/clean/googlelocal_multi_feat.csv
```

---

## Transformer training (irrelevant only)

> We intentionally trained the transformer for **`label_irrelevant`** due to time/compute constraints. `label_ads` and `label_rant_no_visit` remain rules‑based in this version.

**Cross‑validation (sampled)**

```bash
python src/train_transformer.py \
  --data data/clean/googlelocal_multi_labeled.csv \
  --labels label_irrelevant \
  --cv_folds 5 --epochs 1 --max_len 192 --bs 16 \
  --limit 10000 --shuffle_buffer 1 --dataloader_workers 0
```

**Full fine‑tuning (overnight/GPU recommended)**

```bash
# CPU example
python src/train_transformer.py \
  --data data/clean/googlelocal_multi_labeled.csv \
  --labels label_irrelevant \
  --epochs 2 --max_len 224 --bs 16 \
  --train_full --dataloader_workers 0

# GPU speed-up (if available)
python src/train_transformer.py \
  --data data/clean/googlelocal_multi_labeled.csv \
  --labels label_irrelevant \
  --epochs 2 --max_len 224 --bs 16 \
  --fp16 --train_full
```

**Threshold tuning (per‑label)**

```bash
python scripts/tune_thresholds_transformer.py --model_dir models/transformer/distilbert
# -> writes models/thresholds_transformer.json
```

**Ensemble predict + evaluate (large set)**

```bash
python scripts/predict_ensemble.py \
  --data data/clean/googlelocal_multi.csv \
  --policy configs/policy.yaml \
  --model_dir models/transformer/distilbert \
  --thr_path models/thresholds_transformer.json \
  --mode precision \
  --out outputs/preds_ens_transformer.csv

python src/evaluate.py \
  --data data/clean/googlelocal_multi.csv \
  --preds outputs/preds_ens_transformer.csv \
  --policy configs/policy.yaml \
  --dump_errors outputs/errors_large_errors.csv
```

---

## Evaluation & interpretation

- **Small curated set**: human‑labeled; good for sanity checks.
- **Large CA/NY/TX sets**: **pseudo‑labels** from rules. These enable scaling but can inflate metrics for rule‑like patterns.
- Because only `label_irrelevant` is model‑trained, `ads`/`rant` counts are small and rule‑driven; any F1≈1.0 there reflects rule matching more than model generalization.
- Use error dumps (`--dump_errors`) to inspect **false positives/negatives** and to adjust **thresholds** or **rules**.

---

## Roadmap

- Hand‑label 300–500 reviews for a **proper validation set**.
- Train a **multi‑task transformer** for all three labels.
- Add **metadata features** (author history, GPS proximity when permitted) to the model input.
- Hard‑negative mining and data augmentation for robustness.

---

## Built with

- Python, PyTorch, Hugging Face Transformers, scikit‑learn, pandas, numpy
- Streamlit for UI, PyYAML + regex for policies

---

## Demo guidance

- **App**: `streamlit run app/streamlit_app.py`
- **Video outline**:
  1) The problem and policies.  
  2) Rules vs model (and why irrelevant first).  
  3) App demo: upload CSV → run → inspect reasons → export.  
  4) CLI workflow: pseudo‑label → train → threshold → evaluate.  
  5) Results, known limitations, and next steps.

---

## License & acknowledgements

- License: MIT (or Apache‑2.0).  
- Datasets: McAuley Google Local; Kaggle Google Maps Restaurant Reviews.  
- Thanks: dataset maintainers, open‑source authors, and TikTok TechJam 2025 organizers.
