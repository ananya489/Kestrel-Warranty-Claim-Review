# Kestrel Home — Warranty Claim Review

A compact, policy-aware fraud-risk scoring service for Kestrel Home warranty claims.

## What this does

- Trains a small supervised fraud-risk model on the post-1-May-2026 operating regime.
- Uses a **temporal validation**: May 2026 for development, June 2026 for validation.
- Produces `predictions.csv` for every row in `test_unlabelled.csv`.
- Exposes `POST /predict` through FastAPI for one-claim scoring.
- Provides a Streamlit screen that calls the same local model and shows human-readable reasons.
- Produces evidence in `reports/validation.json` and `reports/top40_review_queue.csv`.

The policy changed on 1 May 2026: claims under ₹2,000 became auto-approved without inspection. The model therefore avoids pretending the earlier and later periods are one stationary population.

## Important data handling

The supplied operations policy says customer and operational data must not be published to public repositories or shared outside the engagement team. Keep the supplied CSVs and generated predictions private. The `.gitignore` excludes them from Git.

## Prerequisites

Python 3.10+ recommended.

## Clean-machine setup

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python train_model.py
```

The training command creates:

- `models/fraud_model.joblib`
- `predictions.csv`
- `reports/validation.json`
- `reports/top40_review_queue.csv`

## API

```bash
uvicorn api:app --reload
```

Open the FastAPI docs at `http://127.0.0.1:8000/docs` and call `POST /predict`.

Example payload:

```json
{
  "claim_id": "DEMO-001",
  "submitted_at": "2026-07-10 10:30",
  "partner_id": "SP3219",
  "sku": "KH-MG-03",
  "product_serial": "KH395147108",
  "days_since_purchase": 120,
  "claim_amount_inr": 1500,
  "photo_attached": "N",
  "partner_inspected": "N",
  "claim_description": "motor not running",
  "inspector_note": "",
  "customer_prior_claims": 2,
  "source": "crm"
}
```

Response contains `fraud_score`, `risk_level`, and up to three reasons.

## UI

```bash
streamlit run app.py
```

The screen is intentionally small: enter one claim, assess it, and see the score/reasons.

## Validation

The validation is time-based because the unlabeled set is the most recent period and the May policy change materially changed claim handling.

Current June-2026 holdout evidence from this pack:

- Accuracy: **97.26%** at the validation threshold selected for the stated board KPI.
- ROC-AUC: **0.892**.
- Average precision: **0.279**.
- Top 40 precision: **15.0%** (6 frauds in 40 reviewed claims).
- Observed fraud value in the top 40: **₹7,734**, or **₹193 per investigation** on this holdout.

Accuracy is reported because it is the requested board KPI, but it is not the main operational decision metric in a roughly 3% fraud population. A trivial all-negative classifier can look strong on accuracy. The investigation queue is therefore ranked by score and evaluated separately at the desk's 40-claim capacity.

These are validation estimates, not guarantees on the hidden test outcomes.

## Why not just blame new partners?

New-partner tenure is included as a feature because management asked for it, but the model does not hard-code "new = fraud". The score combines tenure, policy regime, inspection, claim amount relative to product price, prior customer claims, warranty timing and lightweight text signals.

## AI/tool use

Development used ChatGPT and Claude for planning, implementation assistance, debugging and review. The product itself uses a local scikit-learn model and deterministic explanation rules; there is no paid LLM/API dependency.

## Files

```text
app.py
api.py
train_model.py
predictions.csv
requirements.txt
src/
  features.py
  model.py
  reasons.py
models/
reports/
tests/
```

## Limitations

- Only 145 positive labeled outcomes exist in the full labeled history, and only 37 are in the post-May regime. This makes estimates noisy.
- June validation is one month and can be unstable.
- `score` is a ranking probability from a class-weighted logistic model; it should not be interpreted as a perfectly calibrated probability.
- Duplicate claim numbers exist in the historical export because partners can resubmit bounced claims. They are retained as operational events rather than silently deleted.
- Serial numbers are partner-entered and can be messy.
- The model cannot see the hidden investigation outcome for the test set.
