from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

from src.model import score
from src.reasons import explain_claim, risk_band

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODEL = ROOT / "models" / "fraud_model.joblib"
VALIDATION = ROOT / "reports" / "validation.json"
QUEUE = ROOT / "reports" / "top40_review_queue.csv"

st.set_page_config(
    page_title="Kestrel Warranty Review",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Kestrel Warranty Claim Review")
st.caption("Policy-aware fraud risk scoring • local model • no paid API required")

if not MODEL.exists():
    st.error("Model not found. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL)
partners = pd.read_csv(DATA / "partners.csv")
products = pd.read_csv(DATA / "products.csv")

# -----------------------------------------------------------------------------
# Sidebar: single-claim assessment
# -----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Claim")
    claim_id = st.text_input("Claim ID", "DEMO-001")
    submitted_at = st.text_input("Submitted at", "2026-07-10 10:30")
    partner_id = st.selectbox("Partner", sorted(partners.partner_id.unique()))
    sku = st.selectbox("SKU", sorted(products.sku.unique()))
    product_serial = st.text_input("Product serial", "KH123456789")
    days = st.number_input("Days since purchase", 0, 2000, 120)
    amount = st.number_input("Claim amount (₹)", 0.0, 50000.0, 1500.0, step=100.0)
    photo = st.selectbox("Photo attached", ["Y", "N"])
    inspected = st.selectbox("Partner inspected", ["Y", "N"])
    prior = st.number_input("Customer prior claims", 0, 20, 0)
    desc = st.text_area("Claim description", "motor not running")
    note = st.text_area("Inspector note", "")
    source = st.selectbox("Source", ["crm", "legacy_zoho"])

assessed = False

if st.button("Assess claim", type="primary", use_container_width=True):
    row = {
        "claim_id": claim_id,
        "submitted_at": submitted_at,
        "partner_id": partner_id,
        "sku": sku,
        "product_serial": product_serial,
        "days_since_purchase": days,
        "claim_amount_inr": amount,
        "photo_attached": photo,
        "partner_inspected": inspected,
        "claim_description": desc,
        "inspector_note": note,
        "customer_prior_claims": prior,
        "source": source,
    }

    s = float(score(model, pd.DataFrame([row]), partners, products)[0])
    band = risk_band(s)
    assessed = True

    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Fraud score", f"{s:.3f}")
    c2.metric("Risk band", band)

    st.subheader("Why the claim was flagged")
    reasons = explain_claim(row, partners, products, s)
    for reason in reasons:
        st.write("• " + reason)

    if s >= 0.75:
        st.warning("Recommended action: route to investigation before payout.")
    elif s >= 0.45:
        st.info("Recommended action: secondary review if capacity permits.")
    else:
        st.success("Recommended action: normal processing; score is low.")

# -----------------------------------------------------------------------------
# Model evidence
# -----------------------------------------------------------------------------
st.divider()
st.subheader("Model evidence")

try:
    metrics = json.loads(VALIDATION.read_text())

    cols = st.columns(4)
    cols[0].metric("Validation AUC", f"{metrics['roc_auc']:.3f}")
    cols[1].metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    cols[2].metric("Top-40 precision", f"{metrics['top40_precision'] * 100:.1f}%")
    cols[3].metric("Review capacity", "40 / month")

    st.caption(
        "Temporal validation uses June 2026 against May 2026 development data, "
        "matching the post-1-May-2026 policy regime. Accuracy is the stated board KPI; "
        "top-40 performance is the operational metric because the investigation desk can "
        "review at most 40 claims per month."
    )

    e1, e2, e3 = st.columns(3)
    e1.metric("Frauds in top 40", int(metrics["top40_fraud_count"]))
    e2.metric("Observed fraud value", f"₹{metrics['top40_actual_fraud_value']:,.0f}")
    e3.metric(
        "Observed value / investigation",
        f"₹{metrics['top40_actual_fraud_value_per_review']:,.0f}",
    )

    st.caption(
        "The ₹193 figure is observed fraudulent claim value per reviewed claim on the June "
        "holdout; it is not a guaranteed future saving."
    )
except Exception as exc:
    st.warning(f"Validation evidence unavailable: {exc}")

# -----------------------------------------------------------------------------
# Investigation queue
# -----------------------------------------------------------------------------
st.divider()
st.subheader("Investigation queue")
st.write(
    "The model ranks claims by fraud score. Because the investigation desk has a "
    "40-claim monthly capacity, the highest-scoring claims are prioritized rather "
    "than treating the score as proof of fraud."
)

try:
    queue = pd.read_csv(QUEUE).head(40).copy()
    queue.insert(0, "priority", range(1, len(queue) + 1))

    display_cols = [
        "priority",
        "claim_id",
        "partner_id",
        "sku",
        "claim_amount_inr",
        "customer_prior_claims",
        "score",
    ]
    display_cols = [c for c in display_cols if c in queue.columns]

    st.dataframe(
        queue[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "priority": st.column_config.NumberColumn("Priority", format="%d"),
            "claim_amount_inr": st.column_config.NumberColumn("Claim amount (₹)", format="₹%.0f"),
            "score": st.column_config.NumberColumn("Fraud score", format="%.3f"),
        },
    )
except Exception as exc:
    st.info(f"Review queue unavailable: {exc}")

# -----------------------------------------------------------------------------
# Decision guidance and limitations
# -----------------------------------------------------------------------------
st.divider()
st.subheader("Decision guidance")

g1, g2, g3 = st.columns(3)
g1.markdown("**HIGH · ≥ 0.75**\n\nInvestigate before payout.")
g2.markdown("**MEDIUM · 0.45–0.75**\n\nSecondary review if capacity permits.")
g3.markdown("**LOW · < 0.45**\n\nNormal processing unless another control requires review.")

st.subheader("Limitations")
st.markdown(
    """
- Validation is time-based and uses June 2026 as the holdout.
- Fraud is rare, so accuracy alone can be misleading; ranking quality is important for the 40-claim review limit.
- A fraud score prioritizes investigation; it does not prove that a claim is fraudulent.
- New-partner status is a model signal, not a rule that labels new partners as fraudulent.
"""
)

st.caption(
    "Kestrel operations policy: claims under ₹2,000 moved to auto-approval from 1 May 2026; "
    "larger claims still require inspection. Model output should support the investigation desk, "
    "not replace investigation outcomes."
)
