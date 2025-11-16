from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import matplotlib.pyplot as plt

# Support both `python -m src.train` and `python src/train.py` execution styles
try:
    from .features import build_features  # type: ignore
except Exception:  # pragma: no cover - fallback for direct script execution
    import sys as _sys
    import pathlib as _pathlib

    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
    from features import build_features  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "model.joblib"
THRESHOLDS_PATH = DATA_DIR / "user_thresholds_v2.csv"
USER_PROFILES_PATH = DATA_DIR / "user_profiles_v2.csv"
FEATURE_IMPORTANCES_JSON = PROJECT_ROOT / "feature_importances.json"


# Compatibility for Streamlit metrics page
def _find_target_column(df: pd.DataFrame) -> str:
    for col in ["is_fraud", "label", "fraud"]:
        if col in df.columns:
            return col
    raise ValueError("Target column not found. Expected one of: is_fraud, label, fraud")


def _time_aware_split(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    i1 = int(n * 0.70)
    i2 = int(n * 0.85)
    idx = np.arange(n)
    return idx[:i1], idx[i1:i2], idx[i2:]


def _select_features(df: pd.DataFrame) -> List[str]:
    desired = [
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
    return [c for c in desired if c in df.columns]


def load_data() -> pd.DataFrame:
    cleaned = DATA_DIR / "cleaned_transactions.csv"
    enhanced = DATA_DIR / "enhanced_fraud_features_v2.csv"
    path = cleaned if cleaned.exists() else enhanced
    if not path.exists():
        raise FileNotFoundError(f"No dataset found at {cleaned} or {enhanced}")
    df = pd.read_csv(path)
    req = {"timestamp", "user_id", "amount", "receiver_id"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def split_data(df_feat: pd.DataFrame, target_col: str):
    df_sorted = df_feat.sort_values("timestamp").reset_index(drop=True)
    y = df_sorted[target_col].astype(int).values
    feat_cols = _select_features(df_sorted)
    if not feat_cols:
        raise RuntimeError("No derived feature columns found after build_features().")
    X = df_sorted[feat_cols].fillna(0)
    idx_tr, idx_va, idx_te = _time_aware_split(df_sorted)
    return (
        X.iloc[idx_tr], y[idx_tr],
        X.iloc[idx_va], y[idx_va],
        X.iloc[idx_te], y[idx_te],
        df_sorted, feat_cols, (idx_tr, idx_va, idx_te)
    )


def apply_resampling(X_train: pd.DataFrame, y_train: np.ndarray, method: str = "none"):
    method = (method or "none").lower()
    if method == "none":
        return X_train, y_train
    try:
        if method == "random":
            from imblearn.over_sampling import RandomOverSampler
            rs = RandomOverSampler(random_state=42)
            X_res, y_res = rs.fit_resample(X_train, y_train)
            return X_res, y_res
        elif method == "smote":
            from imblearn.over_sampling import SMOTE
            rs = SMOTE(random_state=42)
            X_res, y_res = rs.fit_resample(X_train, y_train)
            return X_res, y_res
        elif method == "smotetomek":
            from imblearn.combine import SMOTETomek
            rs = SMOTETomek(random_state=42)
            X_res, y_res = rs.fit_resample(X_train, y_train)
            return X_res, y_res
        else:
            return X_train, y_train
    except Exception as e:
        print(f"Resampling '{method}' unavailable ({e}); proceeding without resampling.")
        return X_train, y_train


def _make_xgb(scale_pos_weight: float):
    from xgboost import XGBClassifier  # type: ignore

    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )


def train_model(X_train: pd.DataFrame, y_train: np.ndarray) -> object:
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = float(neg / max(1, pos))
    try:
        model = _make_xgb(spw)
        model.fit(X_train, y_train)
        model._model_type = "xgboost"  # type: ignore
        model._feature_names_in_ = list(X_train.columns)  # type: ignore
        return model
    except Exception:
        # Fallback to sklearn GradientBoosting with sample weights
        model = GradientBoostingClassifier(random_state=42)
        sw = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(X_train, y_train, sample_weight=sw)
        model._model_type = "sklearn_gb"  # type: ignore
        model._feature_names_in_ = list(X_train.columns)  # type: ignore
        return model


def calibrate(model: object, X_val: pd.DataFrame, y_val: np.ndarray) -> CalibratedClassifierCV:
    calib = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
    calib.fit(X_val, y_val)
    return calib


def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    try:
        p = model.predict_proba(X)
        if p.ndim == 2 and p.shape[1] > 1:
            return p[:, 1]
        return p.ravel()
    except Exception:
        s = model.decision_function(X)
        return 1 / (1 + np.exp(-s))


def _precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(k, len(scores))
    idx = np.argsort(-scores)[:k]
    return float((y_true[idx] == 1).mean())


def evaluate(model, X_val, y_val, X_test, y_test) -> Dict[str, object]:
    val_scores = _predict_proba(model, X_val)
    test_scores = _predict_proba(model, X_test)

    def metrics(y, s):
        auc = roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")
        prauc = average_precision_score(y, s) if len(np.unique(y)) > 1 else float("nan")
        pred = (s >= 0.5).astype(int)
        prec = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        p50 = _precision_at_k(y, s, 50)
        p100 = _precision_at_k(y, s, 100)
        p200 = _precision_at_k(y, s, 200)
        return dict(auc=auc, pr_auc=prauc, precision=prec, recall=rec, f1=f1, cm=cm, p50=p50, p100=p100, p200=p200)

    return {"val": metrics(y_val, val_scores), "test": metrics(y_test, test_scores), "val_scores": val_scores, "test_scores": test_scores}


def compute_user_thresholds(df_sorted: pd.DataFrame, train_idx: np.ndarray, y_sorted: np.ndarray, model) -> pd.DataFrame:
    X_all = df_sorted[_select_features(df_sorted)].fillna(0)
    train_scores = _predict_proba(model, X_all.iloc[train_idx])
    y_train = y_sorted[train_idx]
    mask_nf = (y_train == 0)
    users = df_sorted.iloc[train_idx]["user_id"].astype(str).values
    df_nf = pd.DataFrame({"user_id": users[mask_nf], "score": train_scores[mask_nf]})
    if df_nf.empty:
        global_thr = float(np.quantile(train_scores, 0.95))
        out = pd.DataFrame({"user_id": df_sorted["user_id"].astype(str).unique(), "threshold_95p": global_thr})
    else:
        th = df_nf.groupby("user_id")["score"].quantile(0.95).rename("threshold_95p").reset_index()
        global_thr = float(np.quantile(df_nf["score"], 0.95))
        all_users = pd.DataFrame({"user_id": df_sorted["user_id"].astype(str).unique()})
        out = all_users.merge(th, on="user_id", how="left")
        out["threshold_95p"] = out["threshold_95p"].fillna(global_thr)
    return out


def _save_user_profiles(df_train: pd.DataFrame) -> None:
    df = df_train.copy().sort_values(["user_id", "timestamp"])  # ensure order
    last_idx = df.groupby("user_id")["timestamp"].idxmax()
    last_info = df.loc[last_idx, ["user_id", "timestamp", "device_id"]].rename(columns={"timestamp": "last_tx_time", "device_id": "last_device"})
    profiles = []
    for uid, grp in df.groupby("user_id"):
        last_time = last_info[last_info["user_id"] == uid]["last_tx_time"].iloc[0]
        window_start = last_time - pd.Timedelta(days=30)
        recent = grp[(grp["timestamp"] >= window_start) & (grp["timestamp"] <= last_time)]
        avg = float(recent["amount"].mean()) if not recent.empty else 0.0
        std = float(recent["amount"].std()) if len(recent) > 1 else 0.0
        cnt = int(len(recent))
        last_recv = grp.sort_values("timestamp")["receiver_id"].tail(5).astype(str).tolist()
        profiles.append({
            "user_id": uid,
            "user_avg_amount_30d": avg,
            "user_std_amount_30d": std,
            "user_tx_count_30d": cnt,
            "last_tx_time": last_time,
            "last_device": last_info[last_info["user_id"] == uid]["last_device"].iloc[0],
            "last_receivers": json.dumps(last_recv),
        })
    out = pd.DataFrame(profiles)
    USER_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(USER_PROFILES_PATH, index=False)


def _extract_feature_importances(estimator, feature_names: List[str]) -> Dict[str, float]:
    # Try calibrated -> list -> underlying estimator
    base = estimator
    try:
        if hasattr(estimator, "calibrated_classifiers_"):
            c = estimator.calibrated_classifiers_[0]
            base = getattr(c, "base_estimator_", getattr(c, "estimator", estimator))
        elif hasattr(estimator, "base_estimator"):
            base = estimator.base_estimator
    except Exception:
        base = estimator

    importances = getattr(base, "feature_importances_", None)
    if importances is None:
        return {name: 0.0 for name in feature_names}
    imp = {name: float(val) for name, val in zip(feature_names, importances)}
    return dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))


def save_outputs(model, thresholds_df: pd.DataFrame, df_train: pd.DataFrame, feat_cols: List[str]) -> None:
    # save model
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    # thresholds
    THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    thresholds_df.to_csv(THRESHOLDS_PATH, index=False)
    print(f"Saved user thresholds to {THRESHOLDS_PATH}")
    # profiles
    _save_user_profiles(df_train)
    print(f"Saved user profiles to {USER_PROFILES_PATH}")
    # feature importances
    imps = _extract_feature_importances(model, feat_cols)
    with open(FEATURE_IMPORTANCES_JSON, "w") as f:
        json.dump(imps, f, indent=2)
    print(f"Saved feature importances to {FEATURE_IMPORTANCES_JSON}")
    # optional plot
    top = list(imps.items())[:20]
    if top:
        names = [k for k, _ in top][::-1]
        vals = [v for _, v in top][::-1]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(names, vals, color="tab:blue")
        ax.set_title("Top 20 feature importances")
        fig.tight_layout()
        out_png = PROJECT_ROOT / "feature_importances.png"
        fig.savefig(out_png)
        plt.close(fig)
        print(f"Saved feature importance plot to {out_png}")


def main(oversample_method: str = None) -> None:
    # 1) Load data
    df = load_data()
    # 2) Build features
    df_feat = build_features(df)
    # 3) Target and split
    target_col = _find_target_column(df_feat)
    X_train, y_train, X_val, y_val, X_test, y_test, df_sorted, feat_cols, (idx_tr, idx_va, idx_te) = split_data(df_feat, target_col)

    print(f"Train size: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f"Class counts (train): neg={(y_train==0).sum()}, pos={(y_train==1).sum()}")

    # 4 & 5) Resampling options for training only
    if oversample_method is None:
        oversample_method = os.getenv("OVERSAMPLE_METHOD", "none")
    X_tr_res, y_tr_res = apply_resampling(X_train, y_train, oversample_method)
    if oversample_method != "none":
        print(f"Resampling applied: {oversample_method} -> new counts: neg={(y_tr_res==0).sum()}, pos={(y_tr_res==1).sum()}")

    # 6) Train model with class imbalance handling
    model = train_model(X_tr_res, y_tr_res)

    # 7) Probability calibration on validation set
    calib = calibrate(model, X_val, y_val)

    # 8) Evaluate
    results = evaluate(calib, X_val, y_val, X_test, y_test)
    val_m = results["val"]
    test_m = results["test"]
    print("\n=== Metrics Summary ===")
    print(f"Validation: AUC={val_m['auc']:.4f}, PR-AUC={val_m['pr_auc']:.4f}, P@50={val_m['p50']:.3f}, P@100={val_m['p100']:.3f}, P@200={val_m['p200']:.3f}")
    print(f"Test:       AUC={test_m['auc']:.4f}, PR-AUC={test_m['pr_auc']:.4f}, P@50={test_m['p50']:.3f}, P@100={test_m['p100']:.3f}, P@200={test_m['p200']:.3f}")
    print("Confusion Matrix (Test, threshold=0.5):")
    print(test_m["cm"])

    # 9) Per-user thresholds (train only, using calibrated model scores)
    thresholds_df = compute_user_thresholds(df_sorted, idx_tr, df_sorted[target_col].astype(int).values, calib)

    # 10) Save all artifacts
    df_train_part = df_sorted.iloc[idx_tr].copy()
    save_outputs(calib, thresholds_df, df_train_part, feat_cols)

    # 11) Top 20 features summary
    imps = _extract_feature_importances(calib, feat_cols)
    print("Top 20 features:")
    for name, val in list(imps.items())[:20]:
        print(f"  {name}: {val:.6f}")


if __name__ == "__main__":
    main()
