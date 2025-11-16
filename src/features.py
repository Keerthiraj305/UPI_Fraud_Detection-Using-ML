import pandas as pd
import numpy as np
from typing import Dict


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features for model training.

    Adds columns:
      - hour, dayofweek, is_night, is_weekend
      - amount_over_user_avg, z_amount_user
      - is_new_receiver, device_changed
      - time_since_last_tx_minutes, velocity_1h, velocity_24h
      - is_large_tx, is_local_transfer

    The function is defensive: if helper columns already exist it will reuse them, otherwise compute
    from available data (e.g., compute per-user averages from the supplied dataframe).
    """
    df = df.copy()

    # parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        raise ValueError("`timestamp` column required to build time features")

    # time based features
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 21)).astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # per-user aggregates: prefer existing summaries if present
    if "user_avg_amount_30d" in df.columns:
        user_avg = df["user_avg_amount_30d"]
    else:
        user_avg = df.groupby("user_id")["amount"].transform("mean")

    if "user_std_amount_30d" in df.columns:
        user_std = df["user_std_amount_30d"].replace(0, np.nan)
    else:
        user_std = df.groupby("user_id")["amount"].transform("std").replace(0, np.nan)

    df["amount_over_user_avg"] = np.where(
        user_avg.fillna(0) == 0,
        0.0,
        df["amount"] / user_avg.replace({0: np.nan}).fillna(0),
    )

    df["z_amount_user"] = (df["amount"] - user_avg.fillna(0)) / user_std.fillna(1)

    # is_new_receiver: if column exists reuse, otherwise compute whether this receiver appeared
    # previously for the same user (ordered by timestamp)
    if "is_new_receiver" not in df.columns:
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        # compute cumulative counts of receiver per user
        grp = df.groupby(["user_id", "receiver_id"]).cumcount()
        # cumcount == 0 means first occurrence of this user-receiver pair
        df["is_new_receiver"] = (grp == 0).astype(int)

    # device_changed: prefer explicit column, else compare to prev_device or previous device per user
    if "device_changed" not in df.columns:
        if "prev_device" in df.columns:
            df["device_changed"] = (df["device_id"].fillna("") != df["prev_device"].fillna("")).astype(int)
        else:
            # compute previous device per user
            df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
            df["prev_device_calc"] = df.groupby("user_id")["device_id"].shift(1)
            df["device_changed"] = (df["device_id"].fillna("") != df["prev_device_calc"].fillna("")).astype(int)
            df.drop(columns=["prev_device_calc"], inplace=True)

    # time_since_last_tx_minutes: prefer existing, else compute using previous timestamp per user
    if "time_since_last_tx_minutes" not in df.columns:
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        prev_ts = df.groupby("user_id")["timestamp"].shift(1)
        df["time_since_last_tx_minutes"] = (df["timestamp"] - prev_ts).dt.total_seconds() / 60.0
        df["time_since_last_tx_minutes"] = df["time_since_last_tx_minutes"].fillna(999999)

    # velocity counts using searchsorted on per-user timestamp arrays
    if "velocity_1h" not in df.columns or "velocity_24h" not in df.columns:
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        df["velocity_1h"] = 0
        df["velocity_24h"] = 0
        for user, group in df.groupby("user_id"):
            times = group["timestamp"].values.astype("datetime64[s]")
            secs = times.astype("int64")
            # compute for each index number of previous timestamps within window
            idx = np.arange(len(secs))
            # for 1 hour (3600s)
            left_1h = np.searchsorted(secs, secs - 3600, side="left")
            counts_1h = idx - left_1h
            # for 24 hours (86400s)
            left_24h = np.searchsorted(secs, secs - 86400, side="left")
            counts_24h = idx - left_24h
            df.loc[group.index, "velocity_1h"] = counts_1h
            df.loc[group.index, "velocity_24h"] = counts_24h

    # is_large_tx: amount > user_avg + 3*user_std (robust to missing std)
    df["is_large_tx"] = ((df["amount"] > (user_avg.fillna(0) + 3 * user_std.fillna(0)))).astype(int)

    # is_local_transfer: compare sender_location and receiver_location if available
    if "sender_location" in df.columns and "receiver_location" in df.columns:
        df["is_local_transfer"] = (df["sender_location"].fillna("") == df["receiver_location"].fillna("")).astype(int)
    else:
        df["is_local_transfer"] = 0

    # receiver_risk_score: compute as fraction of receiver's transactions that were fraud
    if "receiver_risk_score" not in df.columns:
        if "is_fraud" in df.columns or "label" in df.columns:
            fraud_col = "is_fraud" if "is_fraud" in df.columns else "label"
            df = df.sort_values(["receiver_id", "timestamp"]).reset_index(drop=True)
            df["_recv_fraud_count"] = df.groupby("receiver_id")[fraud_col].cumsum().shift(1, fill_value=0)
            df["_recv_tx_count"] = df.groupby("receiver_id").cumcount()
            df["receiver_risk_score"] = np.where(
                df["_recv_tx_count"] > 0,
                df["_recv_fraud_count"] / df["_recv_tx_count"],
                0.0
            )
            df.drop(columns=["_recv_fraud_count", "_recv_tx_count"], inplace=True)
        else:
            df["receiver_risk_score"] = 0.0

    # ensure numeric types where appropriate
    for c in ["amount_over_user_avg", "z_amount_user", "time_since_last_tx_minutes", "velocity_1h", "velocity_24h", "receiver_risk_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df


def build_features_for_input(tx: Dict, user_profiles: pd.DataFrame) -> pd.Series:
    """Build the same derived features for a single transaction `tx`.

    `tx` is a dict with keys like `user_id`, `amount`, `receiver_id`, `timestamp`, `device_id`,
    `sender_location`, `receiver_location`.

    `user_profiles` is a DataFrame keyed by `user_id` that may contain precomputed summaries:
      - `user_avg_amount_30d`, `user_std_amount_30d`, `last_tx_time`, `last_device`,
      - `user_tx_count_30d` (optional)

    The function returns a pd.Series with the derived features.
    """
    # basic fields
    user_id = str(tx.get("user_id"))
    amount = float(tx.get("amount", 0.0))
    receiver_id = str(tx.get("receiver_id", ""))
    ts = pd.to_datetime(tx.get("timestamp"), errors="coerce")
    device = tx.get("device_id")

    # defaults
    user_row = None
    if user_profiles is not None and "user_id" in user_profiles.columns:
        tmp = user_profiles[user_profiles["user_id"].astype(str) == user_id]
        if len(tmp) > 0:
            user_row = tmp.iloc[0]

    # derive time features
    hour = int(ts.hour) if not pd.isna(ts) else 0
    dayofweek = int(ts.dayofweek) if not pd.isna(ts) else 0
    is_night = int((hour < 6) or (hour >= 21))
    is_weekend = int(dayofweek >= 5)

    # user stats
    if user_row is not None and "user_avg_amount_30d" in user_row:
        user_avg = float(user_row.get("user_avg_amount_30d") or 0.0)
    else:
        user_avg = float(user_row.get("user_avg_amount", 0.0)) if user_row is not None else 0.0

    if user_row is not None and "user_std_amount_30d" in user_row:
        user_std = float(user_row.get("user_std_amount_30d") or 0.0)
    else:
        user_std = float(user_row.get("user_std_amount", 0.0)) if user_row is not None else 0.0

    amount_over_user_avg = 0.0 if user_avg == 0 else amount / user_avg
    z_amount_user = (amount - user_avg) / (user_std if user_std > 0 else 1.0)

    # is_new_receiver: best-effort: if user_row has list/set of receivers use it, else fallback to 1
    is_new_receiver = 1
    if user_row is not None:
        if "unique_receivers_30d" in user_row and "user_tx_count_30d" in user_row:
            # cannot deduce whether specific receiver is new from counts; default to 0 (known) if receiver matches
            is_new_receiver = 1
        if "last_receivers" in user_row:
            # if profile stores recent receivers as a list
            last_receivers = user_row.get("last_receivers")
            try:
                is_new_receiver = 0 if receiver_id in last_receivers else 1
            except Exception:
                is_new_receiver = 1

    # device_changed: compare to last_device if present
    device_changed = 1
    if user_row is not None and "last_device" in user_row:
        last_device = user_row.get("last_device")
        device_changed = 0 if str(last_device) == str(device) else 1

    # time_since_last_tx_minutes
    time_since_last_tx_minutes = 999999.0
    if user_row is not None and "last_tx_time" in user_row and not pd.isna(user_row.get("last_tx_time")):
        last = pd.to_datetime(user_row.get("last_tx_time"), errors="coerce")
        if not pd.isna(last) and not pd.isna(ts):
            time_since_last_tx_minutes = (ts - last).total_seconds() / 60.0

    # velocity estimations: approximate using user_tx_count_30d
    velocity_24h = 0
    velocity_1h = 0
    if user_row is not None:
        if "user_tx_count_30d" in user_row and pd.notna(user_row.get("user_tx_count_30d")):
            cnt30 = float(user_row.get("user_tx_count_30d"))
            velocity_24h = cnt30 / 30.0
            velocity_1h = velocity_24h / 24.0
        elif "velocity_24h" in user_row:
            velocity_24h = float(user_row.get("velocity_24h") or 0)
            velocity_1h = float(user_row.get("velocity_1h") or 0)

    # is_large_tx
    is_large_tx = 1 if amount > (user_avg + 3 * user_std) else 0

    # is_local_transfer
    is_local_transfer = 0
    sender_loc = tx.get("sender_location")
    receiver_loc = tx.get("receiver_location")
    if sender_loc is not None and receiver_loc is not None:
        is_local_transfer = 1 if str(sender_loc) == str(receiver_loc) else 0

    # receiver_risk_score: look up from user_profiles if available
    receiver_risk_score = 0.0
    if user_profiles is not None and "receiver_id" in user_profiles.columns and "receiver_risk_score" in user_profiles.columns:
        recv_match = user_profiles[user_profiles["receiver_id"].astype(str) == str(receiver_id)]
        if not recv_match.empty:
            receiver_risk_score = float(recv_match["receiver_risk_score"].iloc[0])

    out = {
        "hour": hour,
        "dayofweek": dayofweek,
        "is_night": is_night,
        "is_weekend": is_weekend,
        "amount_over_user_avg": amount_over_user_avg,
        "z_amount_user": z_amount_user,
        "is_new_receiver": int(is_new_receiver),
        "device_changed": int(device_changed),
        "time_since_last_tx_minutes": float(time_since_last_tx_minutes),
        "velocity_1h": float(velocity_1h),
        "velocity_24h": float(velocity_24h),
        "is_large_tx": int(is_large_tx),
        "is_local_transfer": int(is_local_transfer),
        "receiver_risk_score": float(receiver_risk_score),
    }

    return pd.Series(out)


if __name__ == "__main__":
    print("features module ready — functions: build_features, build_features_for_input")
