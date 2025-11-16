import pandas as pd
from pathlib import Path
from typing import Union
import numpy as np


def load_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Raises FileNotFoundError if the path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    return pd.read_csv(p)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the transactions dataframe.

    - Validates required columns: `user_id`, `amount`, `receiver_id`, `timestamp`.
    - Parses `timestamp` to datetime; missing timestamps are filled with random recent times.
    - Ensures numeric `amount` and string ids.
    """
    required = ["user_id", "amount", "receiver_id", "timestamp"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()

    # Parse timestamp, coerce errors to NaT
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Fill missing timestamps with random times within the last 30 days
    n_missing = int(df["timestamp"].isna().sum())
    if n_missing > 0:
        now = pd.Timestamp.now()
        start = now - pd.Timedelta(days=30)
        total_seconds = int((now - start).total_seconds())
        rng = np.random.default_rng()
        rand_seconds = rng.integers(0, total_seconds + 1, size=n_missing)
        random_ts = [start + pd.Timedelta(seconds=int(s)) for s in rand_seconds]
        df.loc[df["timestamp"].isna(), "timestamp"] = random_ts

    # Ensure amount is numeric; fill NaN with 0
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

    # Cast ids to strings
    df["user_id"] = df["user_id"].astype(str)
    df["receiver_id"] = df["receiver_id"].astype(str)

    return df


def save_clean(df: pd.DataFrame, outpath: Union[str, Path]):
    p = Path(outpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def main() -> None:
    root = Path(__file__).parents[1]
    src_path = root / "data" / "enhanced_fraud_features_v2.csv"
    out_path = root / "data" / "cleaned_transactions.csv"

    print(f"Loading data from: {src_path}")
    df = load_data(src_path)

    print("Cleaning data...")
    df_clean = clean_data(df)

    save_clean(df_clean, out_path)
    print(f"Saved cleaned data to: {out_path}")

    # Basic EDA prints
    n_rows = len(df_clean)
    print(f"Number of rows: {n_rows}")

    if "is_fraud" in df_clean.columns:
        try:
            fraud_ratio = df_clean["is_fraud"].astype(int).mean()
            print(f"Fraud ratio (is_fraud==1): {fraud_ratio:.4f}")
        except Exception:
            print("Column `is_fraud` present but not numeric; skipping fraud ratio.")
    else:
        print("Column `is_fraud` not present; cannot compute fraud ratio.")

    tx_per_user = df_clean.groupby("user_id").size()
    print("Transactions per user summary:")
    print(tx_per_user.describe())
    print("Top 5 users by transaction count:")
    print(tx_per_user.sort_values(ascending=False).head(5))


if __name__ == "__main__":
    main()
