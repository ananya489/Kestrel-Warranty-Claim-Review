from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix
from .features import build_features, matrix


def train_model(train: pd.DataFrame, partners: pd.DataFrame, products: pd.DataFrame, model_path: str):
    labeled = train[train['is_fraud'].notna()].copy()
    labeled['submitted_at'] = pd.to_datetime(labeled['submitted_at'])
    # The May-2026 auto-approval policy changed the data-generating process.
    # Use May-June as the development regime; June is the temporal validation month.
    dev = labeled[(labeled.submitted_at >= '2026-05-01') & (labeled.submitted_at < '2026-06-01')].copy()
    val = labeled[labeled.submitted_at >= '2026-06-01'].copy()
    if dev['is_fraud'].sum() < 5:
        raise ValueError('Too few positive labels in development window')
    Xdev = matrix(build_features(dev, partners, products))
    Xval = matrix(build_features(val, partners, products))
    ydev = dev.is_fraud.astype(int)
    yval = val.is_fraud.astype(int)
    model = Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5, random_state=42))
    ])
    model.fit(Xdev, ydev)
    pv = model.predict_proba(Xval)[:, 1]
    metrics = evaluate(yval, pv, val)

    # Refit on all labeled claims in the post-policy regime for final test scoring.
    final = labeled[labeled.submitted_at >= '2026-05-01'].copy()
    Xfinal = matrix(build_features(final, partners, products))
    yfinal = final.is_fraud.astype(int)
    final_model = Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5, random_state=42))
    ])
    final_model.fit(Xfinal, yfinal)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)
    return metrics, final_model, val


def evaluate(y, p, frame):
    y = np.asarray(y); p = np.asarray(p)
    result = {
        'n_validation': int(len(y)),
        'fraud_rate': float(y.mean()),
        'roc_auc': float(roc_auc_score(y, p)),
        'average_precision': float(average_precision_score(y, p)),
    }
    # Board KPI is accuracy. Report the best threshold on validation, but also
    # report the operational top-40 queue, which matches the 40/month review cap.
    thresholds = np.linspace(0.01, 0.999, 400)
    best = max(thresholds, key=lambda t: accuracy_score(y, p >= t))
    pred = (p >= best).astype(int)
    result.update({
        'accuracy': float(accuracy_score(y, pred)),
        'threshold_for_accuracy': float(best),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y, pred).tolist(),
    })
    k = min(40, len(y))
    idx = np.argsort(-p)[:k]
    result['top40_precision'] = float(y[idx].mean())
    result['top40_fraud_count'] = int(y[idx].sum())
    amounts = frame['claim_amount_inr'].to_numpy()[idx]
    fraud_amount = frame['claim_amount_inr'].to_numpy()[idx] * y[idx]
    result['top40_actual_fraud_value'] = float(fraud_amount.sum())
    result['top40_actual_fraud_value_per_review'] = float(fraud_amount.sum() / k)
    result['top40_claim_value'] = float(amounts.sum())
    return result


def score(model, claims, partners, products):
    X = matrix(build_features(claims, partners, products))
    return model.predict_proba(X)[:, 1]
