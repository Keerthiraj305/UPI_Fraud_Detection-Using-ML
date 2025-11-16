#!/usr/bin/env python3
"""Train Mode C models: XGBoost raw, IsolationForest, meta LR and calibrated meta.

Saves artifacts into `models/` directory.
"""
import json
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import xgboost as xgb
    XGBClassifier = xgb.XGBClassifier
except Exception:
    # fallback to sklearn's hist gradient boosting if xgboost not present
    from sklearn.ensemble import HistGradientBoostingClassifier as XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'enhanced_fraud_features_modec.csv'
MODELS = ROOT / 'models'
MODELS.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA, parse_dates=['timestamp'])
    # ensure target
    if 'is_fraud' not in df.columns:
        raise SystemExit('is_fraud target not found in data')
    return df


def train():
    df = load_data()
    # load feature list if present
    feat_file = MODELS / 'feature_list.csv'
    if feat_file.exists():
        features = pd.read_csv(feat_file)['feature'].tolist()
    else:
        # pick numeric columns except target/timestamps
        features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['is_fraud']]

    X = df[features].fillna(0.0)
    y = df['is_fraud'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train XGBoost/raw classifier
    clf = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
    clf.fit(X_train, y_train)
    # Save raw xgb
    joblib.dump(clf, MODELS / 'xgb_raw.joblib')

    # Train IsolationForest on non-fraud samples
    nonfraud = X_train[y_train == 0]
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    iso.fit(nonfraud)
    joblib.dump(iso, MODELS / 'isolation_forest.joblib')

    # build meta features on train/test
    def meta_features(model_xgb, iso_model, Xdf):
        try:
            xgb_p = model_xgb.predict_proba(Xdf)[:, 1]
        except Exception:
            # fallback if model doesn't support predict_proba
            xgb_p = np.zeros(len(Xdf))
        try:
            iso_s = -iso_model.decision_function(Xdf)
        except Exception:
            iso_s = np.zeros(len(Xdf))
        return np.vstack([xgb_p, iso_s]).T

    meta_X_train = meta_features(clf, iso, X_train)
    meta_X_test = meta_features(clf, iso, X_test)

    meta = LogisticRegression(max_iter=1000)
    meta.fit(meta_X_train, y_train)
    joblib.dump(meta, MODELS / 'meta_lr.joblib')

    # Calibrate final meta
    cal = CalibratedClassifierCV(estimator=meta, cv='prefit', method='isotonic')
    cal.fit(meta_X_test, y_test)
    joblib.dump(cal, MODELS / 'final_calibrated_meta.joblib')

    # Metrics
    probs = cal.predict_proba(meta_X_test)[:, 1]
    metrics = {'test_auc': float(roc_auc_score(y_test, probs)), 'pr_auc': float(average_precision_score(y_test, probs))}
    with open(MODELS / 'train_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # feature importances if xgb has feature_importances_
    try:
        fi = getattr(clf, 'feature_importances_', None)
        if fi is not None:
            df_fi = pd.DataFrame({'feature': features, 'importance': fi})
            df_fi = df_fi.sort_values('importance', ascending=False)
            df_fi.to_csv(MODELS / 'feature_importances.csv', index=False)
    except Exception:
        pass

    print('Training complete. Artifacts saved to', MODELS)


if __name__ == '__main__':
    train()
