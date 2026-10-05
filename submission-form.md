# Kestrel Home — Submission Form

> Fill the final name/contact/repository fields before sending.

### 1. What did you build?
A policy-aware warranty fraud-risk scoring model, FastAPI service and Streamlit review screen. It ranks every test claim by fraud score and provides human-readable reasons for review.

### 2. What score do you expect on the hidden test set, and why?
I expect **roughly 97%+ accuracy** if evaluated as a binary accuracy KPI at the validation-selected threshold, because the June 2026 time-based holdout achieved **97.26% accuracy**. I would not claim a precise hidden-test score because the fraud outcomes are unavailable and the post-policy labeled sample is small. For ranking quality, the more useful validation numbers are ROC-AUC 0.892 and 15% precision in the top 40.

### 3. What evidence shows it works?
Temporal validation on June 2026: 97.26% accuracy, 0.892 ROC-AUC, 0.279 average precision, and 6 frauds in the top 40 reviewed claims. Automated tests also check prediction shape, reference joins and validation thresholds.

### 4. How often does it not work?
The model misses fraud: June recall at the accuracy-oriented threshold was 18.2%. That is why I use it as a prioritisation queue rather than an automatic denial system. The post-May sample is also small, so performance can move as more investigation outcomes arrive.

### 5. What rupees matter?
On the June holdout, the top 40 contained ₹7,734 of observed fraudulent claim value, or ₹193 per investigation. This is an observed validation figure, not guaranteed future savings.

### 6. What did you do about the newer-partner hypothesis?
I included partner tenure but did not hard-code a new-partner rule. I would review partner-level patterns alongside the score rather than treating all new partners as bad.

### 7. What did you deliberately leave out?
No per-claim paid LLM calls, no automatic claim rejection, no complex agent workflow, no causal claim that partner age causes fraud, and no claim that validation results guarantee hidden-test performance.

### 8. AI/tool use and cost
ChatGPT and Claude were used for planning, implementation assistance, debugging and review. The production scoring path is local scikit-learn plus deterministic rules, so **paid API cost is ₹0** for each run.

### 9. Data handling
The supplied policy states that customer and operational data must not be published to public repositories. Raw Kestrel data and predictions should therefore remain private.

### 10. Recording link
`<PASTE GOOGLE DRIVE / APPROVED PRIVATE LINK>`

### 11. Repository / handoff
`<PASTE PRIVATE REPOSITORY OR APPROVED SUBMISSION LOCATION>`

### 12. Time spent
`<ENTER ACTUAL TIME SPENT>`
