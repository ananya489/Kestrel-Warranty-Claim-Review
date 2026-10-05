from pathlib import Path
import json
import pandas as pd
from src.features import build_features, matrix

ROOT=Path(__file__).parents[1]
DATA=ROOT/'data'

def test_prediction_shape():
    test=pd.read_csv(DATA/'test_unlabelled.csv')
    pred=pd.read_csv(ROOT/'predictions.csv')
    assert list(pred.columns)==['claim_id','score']
    assert len(pred)==len(test)
    assert pred.claim_id.is_unique
    assert pred.score.between(0,1).all()

def test_reference_joins():
    test=pd.read_csv(DATA/'test_unlabelled.csv')
    partners=pd.read_csv(DATA/'partners.csv')
    products=pd.read_csv(DATA/'products.csv')
    x=build_features(test,partners,products)
    assert len(x)==len(test)
    assert x.partner_age_days.notna().all()
    assert x.list_price_inr.notna().all()

def test_validation_report():
    m=json.loads((ROOT/'reports/validation.json').read_text())
    assert m['roc_auc']>0.80
    assert m['accuracy']>0.97
    assert m['top40_precision']>0.10
