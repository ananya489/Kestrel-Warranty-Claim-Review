import pandas as pd
from .features import build_features

def explain_claim(row, partners, products, score):
    x = build_features(pd.DataFrame([row]), partners, products).iloc[0]
    reasons=[]
    if x['new_partner_365d']:
        reasons.append('Partner onboarded within the last 12 months')
    if x['under_auto_approval']:
        reasons.append('Claim is below the ₹2,000 post-May auto-approval threshold')
    if x['uninspected']:
        reasons.append('Claim has no partner inspection sign-off')
    if x['customer_prior_claims'] >= 2:
        reasons.append(f"Customer has {int(x['customer_prior_claims'])} prior warranty claims")
    if x['amount_to_list_price'] > 1.0:
        reasons.append('Claim amount exceeds the product list price')
    if x['near_warranty_boundary']:
        reasons.append('Claim is close to the warranty boundary')
    if x['no_photo']:
        reasons.append('No photo was attached')
    if not reasons:
        reasons.append('No single high-risk rule fired; score combines multiple claim signals')
    return reasons[:3]

def risk_band(score):
    if score >= .75: return 'HIGH'
    if score >= .45: return 'MEDIUM'
    return 'LOW'
