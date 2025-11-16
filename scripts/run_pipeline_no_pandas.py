"""Lightweight pipeline that does the required cleaning without pandas.

This is a fallback runner for environments where pandas is not available.
It implements: load_data(path), clean_data(rows), save_clean(rows, outpath), main().
"""
import csv
from pathlib import Path
from datetime import datetime, timedelta
import random
import sys


def load_data(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    return rows


def parse_ts(s):
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def clean_data(rows):
    required = ["user_id", "amount", "receiver_id", "timestamp"]
    if not rows:
        return rows
    header = rows[0].keys()
    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    now = datetime.now()
    start = now - timedelta(days=30)

    cleaned = []
    for r in rows:
        # parse timestamp
        ts = parse_ts(r.get("timestamp", ""))
        if ts is None:
            # random recent
            delta = random.randint(0, int((now - start).total_seconds()))
            ts = start + timedelta(seconds=delta)
        r["timestamp"] = ts.strftime("%Y-%m-%d %H:%M:%S")

        # amount
        a = r.get("amount", "")
        try:
            r["amount"] = float(a) if a not in (None, "") else 0.0
        except Exception:
            r["amount"] = 0.0

        # ids as str
        r["user_id"] = str(r.get("user_id", ""))
        r["receiver_id"] = str(r.get("receiver_id", ""))

        cleaned.append(r)

    return cleaned


def save_clean(rows, outpath):
    p = Path(outpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # write empty file
        with p.open("w", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    root = Path(__file__).parents[1]
    src_path = root / "data" / "enhanced_fraud_features_v2.csv"
    out_path = root / "data" / "cleaned_transactions.csv"

    print("Loading...", src_path)
    rows = load_data(src_path)
    print("Cleaning... rows=", len(rows))
    cleaned = clean_data(rows)
    save_clean(cleaned, out_path)
    print("Saved:", out_path)

    # Basic EDA
    n = len(cleaned)
    print("Number of rows:", n)
    # fraud ratio if is_fraud present
    if cleaned and "is_fraud" in cleaned[0]:
        vals = [r.get("is_fraud", "") for r in cleaned]
        try:
            nums = [int(x) for x in vals if x not in (None,"")]
            ratio = sum(nums)/len(nums) if nums else None
            print("Fraud ratio:", ratio)
        except Exception:
            print("is_fraud present but non-numeric")

    # transactions per user summary
    from collections import Counter
    c = Counter(r.get("user_id", "") for r in cleaned)
    counts = list(c.values())
    counts.sort()
    if counts:
        import statistics
        print("tx count - min/max/mean:", counts[0], counts[-1], statistics.mean(counts))
        top5 = sorted(c.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top5:", top5)


if __name__ == "__main__":
    main()
