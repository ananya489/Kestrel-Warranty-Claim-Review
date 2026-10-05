from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.model import score
from src.reasons import explain_claim, risk_band

ROOT=Path(__file__).parent; DATA=ROOT/'data'; MODEL=ROOT/'models/fraud_model.joblib'
app=FastAPI(title='Kestrel Warranty Fraud Review API', version='1.0')
model=joblib.load(MODEL) if MODEL.exists() else None
partners=pd.read_csv(DATA/'partners.csv'); products=pd.read_csv(DATA/'products.csv')

class Claim(BaseModel):
    claim_id: str='DEMO'
    submitted_at: str
    partner_id: str
    sku: str
    product_serial: str=''
    days_since_purchase: int=0
    claim_amount_inr: float
    photo_attached: str='N'
    partner_inspected: str='N'
    claim_description: str=''
    inspector_note: str=''
    customer_prior_claims: int=0
    source: str='crm'

@app.get('/health')
def health():
    return {'status':'ok','model_loaded':model is not None}

@app.post('/predict')
def predict(claim: Claim):
    if model is None:
        raise HTTPException(status_code=503, detail='Model is not trained. Run python train_model.py first.')
    row=claim.model_dump()
    score_value=float(score(model,pd.DataFrame([row]),partners,products)[0])
    return {'claim_id':claim.claim_id,'fraud_score':round(score_value,6),'risk_level':risk_band(score_value),'reasons':explain_claim(row,partners,products,score_value)}
