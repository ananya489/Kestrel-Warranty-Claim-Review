Kestrel Home — Warranty Claim Review
A lightweight, policy-aware machine learning system for prioritizing potentially fraudulent warranty claims before payout.
What I built
The solution combines:
- Fraud-risk scoring for individual warranty claims
- Partner, product, customer-history and claim features
- Time-based validation
- Top-40 investigation prioritization
- Human-readable risk reasons
- FastAPI prediction endpoint
- Streamlit review interface
- Automated tests
- Predictions for all 2,252 supplied test claims
The system is a decision-support tool, not an automatic fraud verdict.
Business context
Kestrel's investigation desk can review at most 40 claims per month. The model therefore ranks claims by fraud score so the highest-risk claims can be investigated first.
The policy also changed on 1 May 2026: claims below ₹2,000 became auto-approved without inspection, while larger claims still require inspection. This policy regime is included in the modelling rather than treating the full history as one unchanged population.
Validation results
Metric	Result
Validation accuracy	97.26%
ROC-AUC	0.892
Top-40 precision	15.0%
Fraud cases in top 40	6
Investigation capacity	40/month
Test claims scored	2,252


On the June validation holdout, the top-40 queue contained ₹7,734 of observed fraudulent claim value, or approximately ₹193 per reviewed claim. This is an observed validation result, not a guaranteed future saving.
Accuracy is included because it was the requested board KPI. ROC-AUC and top-40 precision are also reported because fraud is imbalanced and the real workflow is capacity-constrained.
Approach
1. Data preparation
Undecided claims are excluded from supervised training. Partner and product reference data are joined by ID, and claim, customer-history, partner and product features are derived.
Key signals include:
- Claim amount
- Days since purchase
- Claim amount relative to product price
- Customer prior claims
- Partner tenure and type
- Partner historical behaviour
- Inspection/photo status
- Product family and warranty duration
- Pre/post-May-2026 policy regime
The "new partner" hypothesis is evaluated as a model signal, not used as a hard fraud rule.
2. Time-based validation
A random split was avoided because the supplied test set represents the most recent claims.
The evaluation follows the real direction of time:
Older claims → TRAIN
Later claims → VALIDATION
Most recent claims → TEST
This provides a more realistic estimate of future performance.
3. Investigation prioritization
Each claim receives a score from 0 to 1:
0 → lower estimated fraud likelihood
1 → higher estimated fraud likelihood
Claims are ranked by score and the highest-risk 40 are prioritized for investigation.
A high score means "investigate first", not "confirmed fraud."
Risk bands
Score	Risk	Suggested action
≥ 0.75	HIGH	Investigate before payout
0.45–0.75	MEDIUM	Secondary review
< 0.45	LOW	Normal processing


Application
FastAPI
A single-record endpoint accepts a claim as JSON and returns:
- Fraud score
- Risk band
- Recommended action
- Human-readable reasons
Streamlit
The review screen provides:
- Single-claim assessment
- Risk score and band
- Explanation
- Recommended action
- Model evidence
- Top-40 investigation queue
- Decision guidance
- Limitations
Project structure
kestrel_warranty_review/
├── app.py
├── api.py
├── train_model.py
├── requirements.txt
├── README.md
├── src/
│   ├── features.py
│   ├── model.py
│   └── reasons.py
├── tests/
│   ├── conftest.py
│   └── test_project.py
└── reports/
    ├── analysis_memo.md
    ├── recording_script.md
    └── validation.json
Private assignment data and generated sensitive artifacts should remain outside a public repository.
Setup
Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Train and validate
python train_model.py
Run tests
pytest -q
Expected:
3 passed
Run FastAPI
uvicorn api:app --reload
Run Streamlit
streamlit run app.py
Prediction output
The required prediction format is:
claim_id,score
WC001,0.0123
WC002,0.7821
WC003,0.0432
The final prediction file contains one score for every test claim.
What was deliberately avoided
- No paid LLM API per claim
- No automatic claim rejection
- No blanket assumption that newer partners are fraudulent
- No random validation split
- No unnecessarily complex agent/LLM pipeline
The core model runs locally and requires no paid API key.
Limitations
- Fraud is highly imbalanced, so accuracy should not be used alone.
- Validation is historical and future fraud behaviour may change.
- The May 2026 policy change can alter the data distribution.
- A model score does not prove fraud.
- The ₹7,734 / ₹193 result is specific to the validation period.
- Human investigation remains the final decision.
Data privacy
The supplied policy states that customer and operational data must not be published to public repositories or shared beyond the engagement team.
Therefore, the original Kestrel CSV/PDF/email files should remain private and should not be committed to a public GitHub repository.
Final recommendation
Use the model as a pre-payout investigation prioritization layer:
Claims
  ↓
Fraud scoring
  ↓
Rank by risk
  ↓
Top 40
  ↓
Investigation
  ↓
Fraud / Not Fraud
  ↓
Track financial value and model performance
The goal is not to replace investigators, but to use their limited review capacity where the model sees the highest expected fraud risk.
