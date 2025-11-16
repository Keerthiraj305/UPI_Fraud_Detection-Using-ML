"""
Tune fusion weights and decision thresholds to improve accuracy/balanced accuracy
without changing code paths. Writes models/decision_config.json that predict_modec reads.

Usage:
  python3 UPI_Fraud_ModeC_Professional_UI/scripts/tune_decision_config.py --sample 3000
"""

from pathlib import Path
import argparse
import json
import random
import pandas as pd
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict_modec import predict_single

DATA = ROOT / 'data' / 'enhanced_fraud_features_modec.csv'
OUT = ROOT / 'models' / 'decision_config.json'


def to_tx(row: pd.Series):
    tx = {
        'user_id': str(row.get('user_id', '0')),
        'amount': float(row.get('amount', 0.0)),
        'hour': int(row['hour']) if 'hour' in row and not pd.isna(row['hour']) else 12,
        'receiver_id': row.get('receiver_id', ''),
        'device_changed': bool(row.get('device_changed', False)),
        'is_new_receiver': bool(row.get('is_new_receiver', False)),
        'velocity_1h': int(row.get('velocity_1h', 0)) if not pd.isna(row.get('velocity_1h', np.nan)) else 0,
        'receiver_risk_score': float(row.get('receiver_risk_score', 0.0)) if not pd.isna(row.get('receiver_risk_score', np.nan)) else 0.0,
        'sender_location': row.get('sender_location', None),
        'receiver_location': row.get('receiver_location', None),
    }
    # Hour fallback from timestamp
    if 'timestamp' in row and not pd.isna(row['timestamp']):
        try:
            ts = pd.to_datetime(row['timestamp'])
            tx['hour'] = int(ts.hour)
        except Exception:
            pass
    return tx


def metrics(y_true, y_pred):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    acc = (y_true == y_pred).mean() if len(y_true) else 0.0
    # balanced accuracy
    pos = y_true == 1
    neg = y_true == 0
    tpr = (y_pred[pos] == 1).mean() if pos.any() else 0.0
    tnr = (y_pred[neg] == 0).mean() if neg.any() else 0.0
    bacc = (tpr + tnr) / 2.0
    return acc, bacc, tpr, tnr


def classify(prob, fraud_thr, susp_thr):
    if prob >= fraud_thr:
        return 1
    elif prob >= susp_thr:
        # Map SUSPICIOUS to positive or negative? For accuracy, treat as positive (fraud-screening)
        return 1
    else:
        return 0


def main(sample_n: int, seed: int, metric: str):
    if not DATA.exists():
        print(f"Dataset not found: {DATA}")
        return 1

    df = pd.read_csv(DATA)
    if 'is_fraud' not in df.columns:
        print("Column 'is_fraud' not found in dataset")
        return 1

    # Sample for speed
    if sample_n and len(df) > sample_n:
        df = df.sample(sample_n, random_state=seed)

    # Prepare labels
    y = df['is_fraud'].astype(int).tolist()

    # Grid to try (small, quick)
    ml_weights = [0.4, 0.5, 0.6, 0.7]
    fraud_thrs = [0.85, 0.90, 0.95]
    susp_thrs = [0.50, 0.55, 0.60]

    # Precompute probabilities with current engine to avoid re-running model per grid
    probs = []
    rows = df.to_dict('records')
    for r in rows:
        tx = to_tx(pd.Series(r))
        prob, decision, details = predict_single(tx, root=ROOT)
        # Store decomposed parts to re-fuse quickly under different weights
        ml_prob = details['ml_scores']['final_ml_prob']
        rule_prob = details['rule_output']['rule_prob']
        probs.append((ml_prob, rule_prob))

    best = None
    for mw in ml_weights:
        rw = 1.0 - mw
        for ft in fraud_thrs:
            for st in susp_thrs:
                preds = []
                for (mlp, rlp) in probs:
                    fused = mw * mlp + rw * rlp
                    preds.append(classify(fused, ft, st))
                acc, bacc, tpr, tnr = metrics(y, preds)
                if metric == 'accuracy':
                    score = (acc, bacc)
                else:
                    score = (bacc, acc)  # default: balanced accuracy first
                if (best is None) or (score > best['score']):
                    best = {
                        'ml_weight': mw,
                        'rule_weight': rw,
                        'fraud_threshold': ft,
                        'suspicious_threshold': st,
                        'acc': acc,
                        'bacc': bacc,
                        'tpr': tpr,
                        'tnr': tnr,
                        'score': score,
                    }

    if best is None:
        print("Tuning failed to find a configuration")
        return 2

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        'ml_weight': best['ml_weight'],
        'rule_weight': best['rule_weight'],
        'fraud_threshold': best['fraud_threshold'],
        'suspicious_threshold': best['suspicious_threshold'],
        'notes': {
            'balanced_accuracy': best['bacc'],
            'accuracy': best['acc'],
            'tpr': best['tpr'],
            'tnr': best['tnr'],
        }
    }
    OUT.write_text(json.dumps(cfg, indent=2))
    print("Saved:", OUT)
    print(json.dumps(cfg, indent=2))
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=2000, help='Sample size for quick tuning')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--metric', type=str, default='balanced', choices=['balanced','accuracy'], help='Primary metric to optimize')
    args = ap.parse_args()
    raise SystemExit(main(args.sample, args.seed, args.metric))
