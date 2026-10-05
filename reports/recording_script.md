# 3-minute recording script

### 0:00–0:25 — What I built
“Hi, this is my Kestrel Warranty Claim Review tool. I built a policy-aware fraud-risk model that ranks claims for the investigation desk, plus a FastAPI endpoint and a small Streamlit screen.”

### 0:25–0:55 — Data / policy change
“The key decision I made was to treat the May 2026 policy change separately. Claims below ₹2,000 became auto-approved without inspection, so I used the post-change period for model development and temporal validation.”

### 0:55–1:25 — Evidence
“On the June 2026 holdout, the model achieved 97.26% accuracy and 0.892 ROC-AUC. More importantly for operations, the top 40 ranked claims contained 6 frauds, or 15% precision.”

### 1:25–1:55 — Rupees
“Those top 40 contained ₹7,734 of observed fraudulent claim value in the holdout, or about ₹193 per investigation. This is a validation estimate, not guaranteed savings. I used this because the investigation desk can only review 40 claims a month.”

### 1:55–2:30 — Product demo
“Here is the one-claim screen. I enter a claim and the service returns a fraud score, risk band and readable reasons. The API exposes the same output at `/predict`, so the UI and service use the same model.”

### 2:30–2:50 — What I changed / discarded
“I tested the idea of simply treating newer partners as the problem, but did not hard-code that assumption. I also avoided paid LLM calls because they were not necessary for this decision and would add cost and another dependency.”

### 2:50–3:00 — Close
“My recommendation is to use the model as a prioritisation queue, not an automatic rejection gate, and retrain as more post-policy investigation outcomes arrive.”
