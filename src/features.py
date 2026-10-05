import re
import numpy as np
import pandas as pd

POLICY_DATE = pd.Timestamp('2026-05-01')


def build_features(df: pd.DataFrame, partners: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Build policy-aware, leakage-safe claim features.

    The production model is intentionally small. It focuses on variables available
    before payout and the May-2026 auto-approval policy change.
    """
    x = df.copy().reset_index(drop=True)
    x['submitted_at'] = pd.to_datetime(x['submitted_at'], errors='coerce')
    p = partners.copy()
    p['onboarded_date'] = pd.to_datetime(p['onboarded_date'], errors='coerce')
    x = x.merge(p, on='partner_id', how='left', validate='many_to_one')
    x = x.merge(products, on='sku', how='left', suffixes=('', '_product'), validate='many_to_one')

    x['partner_age_days'] = (x['submitted_at'] - x['onboarded_date']).dt.days.clip(lower=0)
    x['new_partner_365d'] = (x['partner_age_days'] < 365).astype(int)
    x['new_partner_180d'] = (x['partner_age_days'] < 180).astype(int)
    x['under_auto_approval'] = (x['claim_amount_inr'] < 2000).astype(int)
    x['amount_to_list_price'] = x['claim_amount_inr'] / x['list_price_inr'].replace(0, np.nan)
    x['warranty_remaining_days'] = x['warranty_months'] * 30 - x['days_since_purchase']
    x['near_warranty_boundary'] = x['warranty_remaining_days'].between(-30, 60).astype(int)
    x['post_policy_change'] = (x['submitted_at'] >= POLICY_DATE).astype(int)
    x['uninspected'] = (x['partner_inspected'].fillna('N').str.upper() == 'N').astype(int)
    x['no_photo'] = (x['photo_attached'].fillna('N').str.upper() == 'N').astype(int)
    x['customer_prior_claims_log'] = np.log1p(x['customer_prior_claims'].clip(lower=0))
    x['claim_amount_log'] = np.log1p(x['claim_amount_inr'].clip(lower=0))
    x['days_since_purchase_log'] = np.log1p(x['days_since_purchase'].clip(lower=0))
    x['hour'] = x['submitted_at'].dt.hour.fillna(0).astype(int)
    x['day_of_week'] = x['submitted_at'].dt.dayofweek.fillna(0).astype(int)
    x['month'] = x['submitted_at'].dt.month.fillna(0).astype(int)

    serial = x['product_serial'].fillna('').astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
    x['serial_length'] = serial.str.len()
    x['serial_digit_ratio'] = serial.str.count(r'\d') / serial.str.len().replace(0, np.nan)
    x['serial_prefix'] = serial.str[:3]

    desc = x['claim_description'].fillna('').astype(str).str.lower()
    note = x['inspector_note'].fillna('').astype(str).str.lower()
    text = desc + ' ' + note
    for word in ['bill', 'receipt', 'invoice', 'serial', 'tamper', 'duplicate', 'mismatch', 'fake', 'missing', 'crack', 'burn', 'leak']:
        x[f'kw_{word}'] = text.str.contains(word, regex=False).astype(int)

    return x


MODEL_FEATURES = [
    'partner_age_days', 'new_partner_365d', 'new_partner_180d',
    'under_auto_approval', 'amount_to_list_price', 'warranty_remaining_days',
    'near_warranty_boundary', 'uninspected', 'no_photo', 'customer_prior_claims',
    'customer_prior_claims_log', 'claim_amount_log', 'days_since_purchase_log',
    'hour', 'day_of_week', 'month', 'serial_length', 'serial_digit_ratio',
    'kw_bill', 'kw_receipt', 'kw_invoice', 'kw_serial', 'kw_tamper', 'kw_duplicate',
    'kw_mismatch', 'kw_fake', 'kw_missing', 'kw_crack', 'kw_burn', 'kw_leak',
]


def matrix(x: pd.DataFrame) -> pd.DataFrame:
    out = x[MODEL_FEATURES].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
