from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import joblib

try:
    # When executed as module
    from .features import build_features_for_input  # type: ignore
except Exception:  # pragma: no cover
    # When executed as script
    import sys as _sys
    import pathlib as _pathlib

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
    from features import build_features_for_input  # type: ignore


# Feature list used during training (keep in sync with src/train.py _select_features)
DESIRED_FEATURES: List[str] = [
    "hour",
    "dayofweek",
    "is_night",
    "is_weekend",
    "amount_over_user_avg",
    "z_amount_user",
    "is_new_receiver",
    "device_changed",
    "time_since_last_tx_minutes",
    "velocity_1h",
    "velocity_24h",
    "is_large_tx",
    "is_local_transfer",
    "receiver_risk_score",
]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PREDICTIONS_PATH = DATA_DIR / "predicted_transactions.csv"


def decide_action(prob: float, user_id: str | int, user_thresholds) -> str:
    """Robust decision logic.

    - Load per-user threshold from `user_thresholds` (dict-like or DataFrame):
        threshold = user_thresholds.get(user_id, 0.1)
    - Apply minimum floor: if threshold < 0.1 -> threshold = 0.1
    - Decision order (exact):
        * prob >= 0.95 -> "BLOCK"
        * prob >= threshold -> "SUSPICIOUS"
        * else -> "ALLOW"

    Returns one of: "ALLOW", "SUSPICIOUS", "BLOCK".
    Also prints a debug line with prob and threshold.
    """
    # Default threshold
    threshold = 0.1

    try:
        if user_thresholds is None:
            threshold = 0.1
        # dict-like
        elif isinstance(user_thresholds, dict):
            # Try both string and raw key
            threshold = user_thresholds.get(str(user_id), user_thresholds.get(user_id, 0.1))
            threshold = float(threshold) if threshold is not None and not (isinstance(threshold, float) and np.isnan(threshold)) else 0.1
        # pandas DataFrame
        elif isinstance(user_thresholds, pd.DataFrame):
            df = user_thresholds
            if "user_id" in df.columns and "threshold_95p" in df.columns:
                mask = df["user_id"].astype(str) == str(user_id)
                if mask.any():
                    val = df.loc[mask, "threshold_95p"].iloc[0]
                    threshold = float(val) if pd.notna(val) else 0.1
                else:
                    # No per-user entry -> default
                    threshold = 0.1
            else:
                # Try DataFrame as mapping (index -> value) or single-series
                try:
                    # if it's a single-column DF or Series-like
                    val = df.get(str(user_id), None)
                    if val is None:
                        threshold = 0.1
                    else:
                        threshold = float(val)
                except Exception:
                    threshold = 0.1
        else:
            # Fallback to duck-typed get
            try:
                val = user_thresholds.get(user_id, 0.1)
                threshold = float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else 0.1
            except Exception:
                threshold = 0.1
    except Exception:
        threshold = 0.1

    # Apply minimum floor
    try:
        if threshold is None or (isinstance(threshold, float) and np.isnan(threshold)):
            threshold = 0.1
        if float(threshold) < 0.1:
            threshold = 0.1
        threshold = float(threshold)
    except Exception:
        threshold = 0.1

    # Decision rules (exact order)
    if prob >= 0.95:
        decision = "BLOCK"
    elif prob >= threshold:
        decision = "SUSPICIOUS"
    else:
        decision = "ALLOW"

    # Debug log
    try:
        print(f"[DEBUG] prob={prob:.4f}, threshold={threshold:.4f}, decision={decision}")
    except Exception:
        # Best-effort debug print
        print("[DEBUG] decide_action computed decision")

    return decision


def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    try:
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] > 1:
            return p[:, 1]
        return p.ravel()
    except Exception:
        s = model.decision_function(X)
        return 1 / (1 + np.exp(-s))


def predict_and_save(
    tx: Dict,
    model,
    user_profiles_df: pd.DataFrame,
    user_thresholds_df: pd.DataFrame,
) -> Tuple[float, str, pd.Series]:
    """Compute single-transaction fraud probability and decision, then append a log row.

    Returns (prob, decision, features_series).
    """
    # Build features for input
    feat_series = build_features_for_input(tx, user_profiles_df)

    # Ensure all desired features exist and are numeric
    row = {c: float(feat_series.get(c, 0.0)) for c in DESIRED_FEATURES}
    X = pd.DataFrame([row], columns=DESIRED_FEATURES)

    # Predict probability of fraud
    prob = float(_predict_proba(model, X)[0])

    # Decide action
    user_id = tx.get("user_id")
    decision = decide_action(prob, user_id, user_thresholds_df)

    # Append to predictions CSV
    record = {
        "timestamp": tx.get("timestamp"),
        "user_id": tx.get("user_id"),
        "receiver_id": tx.get("receiver_id"),
        "amount": tx.get("amount"),
        "model_score": prob,
        "decision": decision,
    }
    # include features for traceability
    for k, v in row.items():
        record[f"feat_{k}"] = v

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PREDICTIONS_PATH.exists():
        try:
            exist_df = pd.read_csv(PREDICTIONS_PATH)
        except Exception:
            exist_df = pd.DataFrame()
        out_df = pd.concat([exist_df, pd.DataFrame([record])], ignore_index=True)
    else:
        out_df = pd.DataFrame([record])
    out_df.to_csv(PREDICTIONS_PATH, index=False)

    return prob, decision, feat_series


def load_model(model_path: str | Path = PROJECT_ROOT / "model.joblib"):
    return joblib.load(model_path)
