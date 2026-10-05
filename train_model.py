from pathlib import Path
import json
import pandas as pd
from src.model import train_model, score
from src.features import build_features

ROOT=Path(__file__).parent
DATA=ROOT/'data'; MODELS=ROOT/'models'; REPORTS=ROOT/'reports'
MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
train=pd.read_csv(DATA/'train.csv')
test=pd.read_csv(DATA/'test_unlabelled.csv')
partners=pd.read_csv(DATA/'partners.csv')
products=pd.read_csv(DATA/'products.csv')
metrics, model, val=train_model(train,partners,products,str(MODELS/'fraud_model.joblib'))
with open(REPORTS/'validation.json','w') as f: json.dump(metrics,f,indent=2)
p=score(model,test,partners,products)
out=pd.DataFrame({'claim_id':test.claim_id,'score':p.clip(0,1)})
out.to_csv(ROOT/'predictions.csv',index=False)
# Ranked review queue: operationally, 40 is the desk's monthly capacity.
queue=test.copy(); queue['score']=p; queue=queue.sort_values('score',ascending=False).head(40)
queue.to_csv(REPORTS/'top40_review_queue.csv',index=False)
print(json.dumps(metrics,indent=2))
print(f'Wrote {len(out)} predictions to {ROOT/"predictions.csv"}')
