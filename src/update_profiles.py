from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ENHANCED_PATH = DATA_DIR / "enhanced_fraud_features_v2.csv"
PREDICTED_PATH = DATA_DIR / "predicted_transactions.csv"
OUTPUT_PATH = DATA_DIR / "user_profiles_v2.csv"


def _prepare_base_df(df: pd.DataFrame, *, is_pred: bool = False) -> pd.DataFrame:
    """Normalize columns from either enhanced or predicted sources.

    Ensures columns: user_id, amount, receiver_id, timestamp, is_night, is_weekend, upi_age_days (optional).
    """
    out = df.copy()
    # normalize column names
    for c in ["user_id", "receiver_id"]:
        if c in out.columns:
            out[c] = out[c].astype(str)
    if "amount" in out.columns:
        out["amount"] = pd.to_numeric(out["amount"], errors="coerce")

    # parse timestamp
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    # is_night / is_weekend may be present in enhanced; in predictions, they may come as feat_* columns
    if is_pred:
        if "feat_is_night" in out.columns and "is_night" not in out.columns:
            out["is_night"] = pd.to_numeric(out["feat_is_night"], errors="coerce")
        if "feat_is_weekend" in out.columns and "is_weekend" not in out.columns:
            out["is_weekend"] = pd.to_numeric(out["feat_is_weekend"], errors="coerce")

    # derive if missing
    if "is_night" not in out.columns or out["is_night"].isna().all():
        if "timestamp" in out.columns:
            hours = out["timestamp"].dt.hour
            out["is_night"] = ((hours < 6) | (hours >= 21)).astype(float)
        else:
            out["is_night"] = np.nan

    if "is_weekend" not in out.columns or out["is_weekend"].isna().all():
        if "timestamp" in out.columns:
            out["is_weekend"] = (out["timestamp"].dt.dayofweek >= 5).astype(float)
        else:
            out["is_weekend"] = np.nan

    # ensure expected columns exist
    for col in ["user_id", "amount", "receiver_id", "timestamp", "is_night", "is_weekend"]:
        if col not in out.columns:
            out[col] = np.nan

    # upi_age_days may exist in enhanced; keep if present
    if "upi_age_days" in out.columns:
        out["upi_age_days"] = pd.to_numeric(out["upi_age_days"], errors="coerce")

    return out[[c for c in ["user_id", "amount", "receiver_id", "timestamp", "is_night", "is_weekend", "upi_age_days"] if c in out.columns]]


def rebuild_user_profiles() -> pd.DataFrame:
    """Rebuild user profiles from enhanced dataset and predicted transactions, then save CSV."""
    frames = []
    if ENHANCED_PATH.exists():
        enh = pd.read_csv(ENHANCED_PATH)
        frames.append(_prepare_base_df(enh, is_pred=False))
    if PREDICTED_PATH.exists():
        pred = pd.read_csv(PREDICTED_PATH)
        frames.append(_prepare_base_df(pred, is_pred=True))
    if not frames:
        raise FileNotFoundError(f"No input data found: {ENHANCED_PATH} nor {PREDICTED_PATH}")

    base = pd.concat(frames, ignore_index=True)
    base = base.dropna(subset=["user_id"]).copy()

    # Aggregations per user
    agg_dict = {
        "amount": ["count", "mean", "std"],
        "receiver_id": pd.Series.nunique,
        "is_night": "mean",
        "is_weekend": "mean",
    }
    if "upi_age_days" in base.columns:
        agg_dict["upi_age_days"] = "max"  # use max observed age

    grouped = base.groupby("user_id").agg(agg_dict)
    # flatten MultiIndex columns
    grouped.columns = [
        "user_tx_count" if (c == ("amount", "count")) else
        "user_avg_amount" if (c == ("amount", "mean")) else
        "user_std_amount" if (c == ("amount", "std")) else
        "unique_receivers" if (c[0] == "receiver_id") else
        "pct_night" if (c[0] == "is_night") else
        "pct_weekend" if (c[0] == "is_weekend") else
        "upi_age_days" if (c[0] == "upi_age_days") else
        "_".join([str(x) for x in c])
        for c in grouped.columns.to_list()
    ]

    profiles = grouped.reset_index()

    # cleanup: fill NaNs
    for col in ["user_tx_count", "unique_receivers"]:
        if col in profiles.columns:
            profiles[col] = profiles[col].fillna(0).astype(int)
    for col in ["user_avg_amount", "user_std_amount", "pct_night", "pct_weekend", "upi_age_days"]:
        if col in profiles.columns:
            profiles[col] = pd.to_numeric(profiles[col], errors="coerce").fillna(0.0)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(OUTPUT_PATH, index=False)
    return profiles


def main() -> None:
    prof = rebuild_user_profiles()
    print(f"Rebuilt profiles for {len(prof)} users -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
