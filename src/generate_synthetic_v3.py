"""Generate a leakage-free synthetic UPI fraud dataset (v3).

This script creates:
  - data/enhanced_fraud_features_v3.csv
  - data/user_profiles_v3.csv
  - data/user_thresholds_v3.csv
  - data/predicted_transactions_v3.csv (empty)

Functions:
  load_and_clean()
  engineer_features()
  generate_fraud_labels_no_leakage()
  build_user_profiles()
  train_small_model_and_thresholds()
  save_outputs()
  main()

All labels are generated from hidden triggers independent of the trainable features.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_and_clean(path: Path | str = DATA_DIR / "transactions.csv", n_rows: int = 5000) -> pd.DataFrame:
    """Load raw transactions or synthesize if missing.

    Returns a DataFrame with columns:
      timestamp, user_id, amount, receiver_id, device_id, sender_location, receiver_location, transaction_type
    """
    p = Path(path)
    if p.exists():
        df = pd.read_csv(p)
    else:
        # Synthesize a base raw transactions dataset
        rng = np.random.default_rng(42)
        users = [f"u_{i}" for i in range(10)]
        cities = [
            "Mumbai",
            "Delhi",
            "Bengaluru",
            "Hyderabad",
            "Chennai",
            "Kolkata",
            "Pune",
            "Ahmedabad",
        ]
        tx_types = ["P2P", "Merchant", "Bill", "Other"]

        base = []
        start_ts = pd.Timestamp("2025-01-01")
        for i in range(n_rows):
            user = rng.choice(users)
            # Random walk of time
            ts = start_ts + pd.Timedelta(days=int(rng.integers(0, 365)), hours=int(rng.integers(0, 24)), minutes=int(rng.integers(0, 60)))
            amount = float(round(10 ** (rng.uniform(1.0, 4.0)), 2))  # between ~10 and 10000
            receiver = f"r_{rng.integers(0, 200)}"
            device = f"device_{rng.integers(0,50)}"
            sender_loc = rng.choice(cities)
            # small bias for receiver location
            receiver_loc = rng.choice(cities) if rng.random() < 0.7 else sender_loc
            tx_type = rng.choice(tx_types)
            base.append(
                {
                    "timestamp": ts.isoformat(sep=" "),
                    "user_id": user,
                    "amount": amount,
                    "receiver_id": receiver,
                    "device_id": device,
                    "sender_location": sender_loc,
                    "receiver_location": receiver_loc,
                    "transaction_type": tx_type,
                }
            )
        df = pd.DataFrame(base).sort_values("timestamp").reset_index(drop=True)

    # Clean fields
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.Timestamp.now()

    df["user_id"] = df["user_id"].astype(str)
    df["receiver_id"] = df["receiver_id"].astype(str)
    df["device_id"] = df.get("device_id", pd.Series([f"device_{i%50}" for i in range(len(df))])).astype(str)
    df["transaction_type"] = df.get("transaction_type", "P2P").astype(str)
    df["sender_location"] = df.get("sender_location", "Unknown").astype(str)
    df["receiver_location"] = df.get("receiver_location", "Unknown").astype(str)

    # Force exactly 10 users u_0..u_9
    users_fixed = [f"u_{i}" for i in range(10)]
    rng = np.random.default_rng(123)
    df["user_id"] = [rng.choice(users_fixed) for _ in range(len(df))]

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features specified in the requirements.

    Note: receiver_risk_score is synthesized per receiver independent of labels to avoid leakage.
    """
    df = df.copy().sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 23]).astype(int)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # Per-user averages
    user_stats = df.groupby("user_id")["amount"].agg(["mean", "std"]).rename(columns={"mean": "user_mean", "std": "user_std"})
    df = df.join(user_stats, on="user_id")
    df["user_std"] = df["user_std"].fillna(0.0) + 1e-6
    df["amount_over_user_avg"] = (df["amount"] - df["user_mean"]) / (df["user_mean"].replace(0, 1))
    df["z_amount_user"] = (df["amount"] - df["user_mean"]) / df["user_std"]

    # is_new_receiver, device_changed, time_since_last_tx_minutes
    df["is_new_receiver"] = 0
    df["device_changed"] = 0
    df["time_since_last_tx_minutes"] = np.nan

    last_seen = {}
    last_device = {}
    for idx, row in df.iterrows():
        u = row["user_id"]
        r = row["receiver_id"]
        t = row["timestamp"]
        dev = row["device_id"]

        # new receiver
        if u not in last_seen:
            df.at[idx, "is_new_receiver"] = 1
        else:
            df.at[idx, "is_new_receiver"] = 0 if r in last_seen[u]["receivers"] else 1

        # device changed
        if u not in last_device:
            df.at[idx, "device_changed"] = 0
        else:
            df.at[idx, "device_changed"] = 0 if last_device[u] == dev else 1

        # time since last tx
        if u in last_seen:
            prev_ts = last_seen[u]["last_ts"]
            df.at[idx, "time_since_last_tx_minutes"] = (t - prev_ts).total_seconds() / 60.0
        else:
            df.at[idx, "time_since_last_tx_minutes"] = np.nan

        # update trackers
        if u not in last_seen:
            last_seen[u] = {"receivers": set([r]), "last_ts": t}
        else:
            last_seen[u]["receivers"].add(r)
            last_seen[u]["last_ts"] = t
        last_device[u] = dev

    df["time_since_last_tx_minutes"] = df["time_since_last_tx_minutes"].fillna(99999.0)

    # velocity_1h and velocity_24h: number of tx by same user in previous window
    df["velocity_1h"] = 0
    df["velocity_24h"] = 0
    for u, group in df.groupby("user_id"):
        times = group["timestamp"].values
        idxs = group.index.values
        for i, t in enumerate(times):
            t0 = pd.to_datetime(t)
            mask_1h = (group["timestamp"] < t0) & (group["timestamp"] >= t0 - pd.Timedelta(hours=1))
            mask_24h = (group["timestamp"] < t0) & (group["timestamp"] >= t0 - pd.Timedelta(hours=24))
            df.at[idxs[i], "velocity_1h"] = int(mask_1h.sum())
            df.at[idxs[i], "velocity_24h"] = int(mask_24h.sum())

    # is_large_tx and is_local_transfer
    df["is_large_tx"] = (df["amount"] > (df["user_mean"] + 3 * df["user_std"])).astype(int)
    df["is_local_transfer"] = (df["sender_location"] == df["receiver_location"]).astype(int)

    # receiver_risk_score: random per receiver, independent of labels
    rng = np.random.default_rng(2025)
    recv_unique = df["receiver_id"].unique()
    recv_risk = {r: float(rng.random()) for r in recv_unique}
    df["receiver_risk_score"] = df["receiver_id"].map(recv_risk).fillna(0.0)

    # Drop temp cols
    df = df.drop(columns=["user_mean", "user_std"], errors="ignore")

    # Final feature set ordering (ensure all exist)
    feat_cols = [
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
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0

    return df


def generate_fraud_labels_no_leakage(df: pd.DataFrame, target_fraud_rate_min: float = 0.05, target_fraud_rate_max: float = 0.12) -> pd.DataFrame:
    """Generate fraud labels from hidden triggers that are not functions of the engineered features.

    Ensures fraud ratio in [min, max].
    """
    df = df.copy()
    rng = np.random.default_rng(999)

    n = len(df)

    # Hidden triggers
    fraud_trigger_1 = rng.random(n) < 0.02  # random rare events

    # hidden pattern by user_id: assign a small set of users with higher hidden propensity
    high_risk_users = set(rng.choice(df["user_id"].unique(), size=2, replace=False))
    fraud_trigger_2 = df["user_id"].isin(high_risk_users).values

    # rare merchant (receiver) groups
    recv_group = pd.Series(df["receiver_id"].astype(str).str.extract(r"r_(\d+)")[0].fillna("0").astype(int))
    rare_recv_ids = set(recv_group[recv_group % 50 == 0].index.tolist())
    fraud_trigger_3 = recv_group % 97 == 0  # arbitrary rare pattern

    # Compose noisy probabilistic logic
    base_prob = rng.random(n) * 0.4
    hidden_flags = (fraud_trigger_1.astype(int) + fraud_trigger_2.astype(int) + fraud_trigger_3.astype(int))
    hidden_prob = rng.random(n) * 0.3 + hidden_flags * 0.4
    risk_prob = base_prob + hidden_prob

    labels = (risk_prob > 0.75).astype(int)

    # Adjust to ensure fraud ratio between bounds by scaling threshold if needed
    fraud_rate = labels.mean()
    if fraud_rate < target_fraud_rate_min or fraud_rate > target_fraud_rate_max:
        # adaptively adjust cutoff using percentiles of risk_prob
        low_q = int(100 * (1 - target_fraud_rate_max))
        high_q = int(100 * (1 - target_fraud_rate_min))
        cutoff = float(np.percentile(risk_prob, high_q))
        labels = (risk_prob >= cutoff).astype(int)

    # Final check, if still out of bounds, randomly flip labels to meet lower bound
    fraud_rate = labels.mean()
    if fraud_rate < target_fraud_rate_min:
        need = int(np.ceil(target_fraud_rate_min * n - labels.sum()))
        zeros = np.where(labels == 0)[0]
        flip = rng.choice(zeros, size=min(need, len(zeros)), replace=False)
        labels[flip] = 1

    if labels.mean() > target_fraud_rate_max:
        # lower by flipping some ones to zero
        need = int(np.ceil(labels.sum() - target_fraud_rate_max * n))
        ones = np.where(labels == 1)[0]
        flip = rng.choice(ones, size=min(need, len(ones)), replace=False)
        labels[flip] = 0

    df["is_fraud"] = labels
    return df


def build_user_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute user profiles v3 using 30-day windows relative to max timestamp.

    Ensures pct_night_tx_30d and pct_weekend_tx_30d are never zero by injecting small non-zero values.
    """
    df = df.copy()
    max_ts = df["timestamp"].max()
    window_start = max_ts - pd.Timedelta(days=30)
    recent = df[df["timestamp"] >= window_start]

    profiles = []
    for u, g in recent.groupby("user_id"):
        cnt = len(g)
        avg = float(g["amount"].mean()) if cnt > 0 else 0.0
        std = float(g["amount"].std()) if cnt > 1 else 0.0
        uniq_recv = g["receiver_id"].nunique()
        pct_night = float(g["is_night"].mean()) if cnt > 0 else 0.0
        pct_weekend = float(g["is_weekend"].mean()) if cnt > 0 else 0.0
        upi_age_days = (max_ts - g["timestamp"].min()).days if cnt > 0 else 0

        # ensure non-zero small values
        if pct_night == 0.0:
            pct_night = 0.02
        if pct_weekend == 0.0:
            pct_weekend = 0.01

        profiles.append(
            {
                "user_id": u,
                "user_tx_count_30d": cnt,
                "user_avg_amount_30d": round(avg, 2),
                "user_std_amount_30d": round(std, 2),
                "unique_receivers_30d": int(uniq_recv),
                "pct_night_tx_30d": pct_night,
                "pct_weekend_tx_30d": pct_weekend,
                "upi_age_days": int(upi_age_days),
            }
        )

    profiles_df = pd.DataFrame(profiles)
    # Ensure all users present
    all_users = sorted(df["user_id"].unique())
    for u in all_users:
        if u not in profiles_df["user_id"].values:
            profiles_df = pd.concat([profiles_df, pd.DataFrame([{"user_id": u,
                                                                 "user_tx_count_30d": 0,
                                                                 "user_avg_amount_30d": 0.0,
                                                                 "user_std_amount_30d": 0.0,
                                                                 "unique_receivers_30d": 0,
                                                                 "pct_night_tx_30d": 0.02,
                                                                 "pct_weekend_tx_30d": 0.01,
                                                                 "upi_age_days": 0}])], ignore_index=True)

    return profiles_df.sort_values("user_id").reset_index(drop=True)


def train_small_model_and_thresholds(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Train GradientBoostingClassifier on features and compute per-user 95th percentile threshold of NON-FRAUD scores.

    Returns DataFrame with columns: user_id, threshold_95p
    """
    X = df[feature_cols].fillna(0)
    y = df["is_fraud"].astype(int)

    # Split train/test but we will only use train predictions for thresholds
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.3, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )

    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predict probabilities on train set only
    p_train = clf.predict_proba(X_train)[:, 1]

    # map index to user_id
    train_idx = idx_train
    train_users = df.loc[train_idx, "user_id"].values

    thresholds = []
    df_train_probs = pd.DataFrame({"index": train_idx, "user_id": train_users, "prob": p_train})

    for u, g in df_train_probs.groupby("user_id"):
        nonfraud_probs = g.loc[df.loc[g["index"], "is_fraud"].values == 0, "prob"].values
        if len(nonfraud_probs) == 0:
            thr = 0.1
        else:
            thr = float(np.percentile(nonfraud_probs, 95))
            if np.isnan(thr):
                thr = 0.1
        thr = max(thr, 0.1)
        thresholds.append({"user_id": u, "threshold_95p": thr})

    thresholds_df = pd.DataFrame(thresholds)
    return thresholds_df, clf


def save_outputs(df: pd.DataFrame, profiles: pd.DataFrame, thresholds: pd.DataFrame):
    df_out = df.copy()
    # Save enhanced features
    enhanced_path = DATA_DIR / "enhanced_fraud_features_v3.csv"
    df_out.to_csv(enhanced_path, index=False)

    profiles_path = DATA_DIR / "user_profiles_v3.csv"
    profiles.to_csv(profiles_path, index=False)

    thresholds_path = DATA_DIR / "user_thresholds_v3.csv"
    thresholds.to_csv(thresholds_path, index=False)

    # Empty predictions file
    preds_path = DATA_DIR / "predicted_transactions_v3.csv"
    pd.DataFrame(columns=["timestamp", "user_id", "receiver_id", "amount", "model_score", "decision"]).to_csv(preds_path, index=False)

    print(f"Saved: {enhanced_path}, {profiles_path}, {thresholds_path}, {preds_path}")


def main():
    print("Generating leakage-free synthetic dataset v3...")
    raw = load_and_clean()
    df_feat = engineer_features(raw)

    # Generate labels without leakage
    df_labeled = generate_fraud_labels_no_leakage(df_feat)

    # Build user profiles v3
    profiles = build_user_profiles(df_labeled)

    # Feature columns to use for model
    feature_cols = [
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

    thresholds_df, helper_model = train_small_model_and_thresholds(df_labeled, feature_cols)

    save_outputs(df_labeled, profiles, thresholds_df)

    # Summary
    fraud_ratio = df_labeled["is_fraud"].mean()
    print(f"Fraud ratio: {fraud_ratio:.4f}")

    per_user = df_labeled.groupby("user_id")["is_fraud"].mean().sort_values(ascending=False)
    print("Fraud distribution per user (top):")
    print(per_user.head(10))

    print("Sample rows:")
    print(df_labeled.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
