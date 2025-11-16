import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1]
DATA_DIR = ROOT / "data"

# Ensure project root is on sys.path so `import src...` works when running Streamlit from this folder
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import project utilities after path fix
from src.predict_utils import predict_and_save, DESIRED_FEATURES, load_model as _load_model  # type: ignore
from src.predict import explain_shap  # type: ignore

# Caching loaders
@st.cache_data(show_spinner=False)
def load_enhanced() -> pd.DataFrame:
    p = DATA_DIR / "enhanced_fraud_features_v2.csv"
    if p.exists():
        df = pd.read_csv(p)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    p = DATA_DIR / "predicted_transactions.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_user_profiles() -> pd.DataFrame:
    p = DATA_DIR / "user_profiles_v2.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_thresholds() -> pd.DataFrame:
    p = DATA_DIR / "user_thresholds_v2.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model():
    return _load_model(ROOT / "model.joblib")


def home_location_for_user(enhanced: pd.DataFrame, user_id: str) -> str:
    if enhanced.empty:
        return ""
    sub = enhanced[enhanced.get("user_id", "").astype(str) == str(user_id)]
    if "sender_location" in sub.columns and not sub.empty:
        return sub["sender_location"].astype(str).mode().iloc[0]
    return ""


def last_device_for_user(enhanced: pd.DataFrame, user_id: str) -> str:
    if enhanced.empty:
        return ""
    sub = enhanced[enhanced.get("user_id", "").astype(str) == str(user_id)]
    if "timestamp" in sub.columns and "device_id" in sub.columns and not sub.empty:
        sub = sub.sort_values("timestamp")
        return str(sub["device_id"].iloc[-1])
    return ""


def page_score_transaction():
    st.title("UPI Fraud Scoring")

    profiles = load_user_profiles()
    thresholds = load_thresholds()
    enhanced = load_enhanced()

    if profiles.empty:
        st.warning("User profiles not found. Generate via training or update utility.")

    users = profiles["user_id"].astype(str).tolist() if "user_id" in profiles.columns else []
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Transaction Input")
        user_id = st.selectbox("User", users) if users else st.text_input("User")

        # Defaults from profile if available
        prof_row = profiles[profiles["user_id"].astype(str) == str(user_id)].iloc[0] if (not profiles.empty and user_id in users) else None
        avg_amt = float(prof_row.get("user_avg_amount_30d", prof_row.get("user_avg_amount", 1000.0))) if prof_row is not None else 1000.0
        std_amt = float(prof_row.get("user_std_amount_30d", prof_row.get("user_std_amount", 100.0))) if prof_row is not None else 100.0
        amount = st.number_input("Amount", min_value=0.0, value=float(round(avg_amt, 2)))

        hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=12)
        receiver_id = st.text_input("Receiver ID", value="merchant_1")
        # Locations
        home_loc = home_location_for_user(enhanced, user_id)
        receiver_location = st.text_input("Receiver Location", value=home_loc)
        # Device
        default_device = str(prof_row.get("last_device", last_device_for_user(enhanced, user_id))) if prof_row is not None else last_device_for_user(enhanced, user_id)
        device_id = st.text_input("Device ID", value=default_device or "device_1")
        # Transaction type options
        tx_types = ["P2P", "Merchant", "Bill", "Other"]
        if not enhanced.empty and "transaction_type" in enhanced.columns:
            tx_types = sorted([str(x) for x in enhanced["transaction_type"].dropna().unique().tolist()]) or tx_types
        transaction_type = st.selectbox("Transaction Type", tx_types)

        predict_clicked = st.button("Predict")

    with right:
        st.subheader("User Profile Summary")
        if prof_row is not None:
            upi_age = prof_row.get("upi_age_days", np.nan)
            total_tx = int(prof_row.get("user_tx_count_30d", prof_row.get("user_tx_count", 0)))
            uniq_recv = int(prof_row.get("unique_receivers_30d", prof_row.get("unique_receivers", 0)))
            avg30 = float(prof_row.get("user_avg_amount_30d", prof_row.get("user_avg_amount", np.nan)))
            std30 = float(prof_row.get("user_std_amount_30d", prof_row.get("user_std_amount", np.nan)))
            pct_night = float(prof_row.get("pct_night", np.nan))
            pct_weekend = float(prof_row.get("pct_weekend", np.nan))
            last_dev = str(prof_row.get("last_device", last_device_for_user(enhanced, user_id)))

            st.write(
                f"Home Location: {home_loc or '-'}\n\n"
                f"UPI Age (days): {upi_age if not pd.isna(upi_age) else '-'}\n\n"
                f"Avg Tx Amount (30d): {avg30 if not pd.isna(avg30) else '-'}\n\n"
                f"Std Amount: {std30 if not pd.isna(std30) else '-'}\n\n"
                f"Total Tx (30d): {total_tx}\n\n"
                f"Unique Receivers (30d): {uniq_recv}\n\n"
                f"% Night: {round(100*pct_night,1) if not pd.isna(pct_night) else '-'}\n\n"
                f"% Weekend: {round(100*pct_weekend,1) if not pd.isna(pct_weekend) else '-'}\n\n"
                f"Last Known Device: {last_dev or '-'}"
            )
        else:
            st.info("No profile for this user.")

    if 'predict_clicked' in locals() and predict_clicked:
        import datetime as dt

        model = load_model()
        ts = pd.Timestamp.now().replace(hour=int(hour), minute=0, second=0, microsecond=0)
        tx = {
            "user_id": user_id,
            "amount": amount,
            "receiver_id": receiver_id,
            "timestamp": ts.isoformat(sep=" "),
            "device_id": device_id,
            "sender_location": home_loc,
            "receiver_location": receiver_location,
            "transaction_type": transaction_type,
        }
        prob, decision, feat_series = predict_and_save(tx, model, profiles, thresholds)

        st.markdown(f"**Fraud Probability:** {prob:.4f}")
        st.markdown(f"**Decision:** {decision}")

        st.subheader("Features Used")
        ordered = {k: feat_series.get(k, np.nan) for k in DESIRED_FEATURES}
        st.dataframe(pd.DataFrame([ordered]).T.rename(columns={0: "value"}))

        st.subheader("Explanation (SHAP-like)")
        contrib = explain_shap(pd.Series(ordered), model, list(ordered.keys()))
        # Bar chart
        top = contrib.abs().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(top.index[::-1], top.values[::-1], color="tab:blue")
        ax.set_xlabel("|contribution|")
        ax.set_title("Top feature contributions")
        st.pyplot(fig)


def page_enhanced_csv():
    st.title("Enhanced CSV")
    df = load_enhanced()
    if df.empty:
        st.info("No enhanced data found.")
        return
    st.download_button("Download CSV", data=df.to_csv(index=False), file_name="enhanced_fraud_features_v2.csv")
    st.dataframe(df.head(500))


def page_predictions_history():
    st.title("Predictions History")
    df = load_predictions()
    if df.empty:
        st.info("No predictions yet.")
        return
    profiles = load_user_profiles()
    users = sorted(df["user_id"].astype(str).unique().tolist()) if "user_id" in df.columns else []
    user = st.selectbox("Filter by user", options=["(All)"] + users)
    show = df.copy()
    if user != "(All)":
        show = show[show["user_id"].astype(str) == user]
    st.download_button("Download filtered CSV", data=show.to_csv(index=False), file_name="predicted_transactions_filtered.csv")
    st.dataframe(show)


def main():
    st.set_page_config(page_title="UPI Fraud Prototype", layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Score Transaction", "Enhanced CSV", "Predictions History", "Model Metrics"])
    if page == "Score Transaction":
        page_score_transaction()
    elif page == "Enhanced CSV":
        page_enhanced_csv()
    elif page == "Predictions History":
        page_predictions_history()
    else:
        page_model_metrics()


@st.cache_data(show_spinner=False)
def _compute_model_metrics():
    from sklearn.metrics import roc_auc_score, confusion_matrix
    from src.train import _time_aware_split as _split  # type: ignore
    from src.train import _find_target_column as _target  # type: ignore
    from src.features import build_features  # type: ignore

    model = load_model()
    # Use exact feature names from model
    if hasattr(model, 'feature_names_in_'):
        feat_cols = list(model.feature_names_in_)
    else:
        feat_cols = DESIRED_FEATURES
    
    df = load_enhanced()
    if df.empty:
        return None
    try:
        df_feat = build_features(df)
    except Exception:
        return None
    try:
        target_col = _target(df_feat)
    except Exception:
        return None
    y = df_feat[target_col].astype(int).values
    train_idx, val_idx, test_idx = _split(df_feat)
    
    # Ensure all required features exist
    missing = [c for c in feat_cols if c not in df_feat.columns]
    if missing:
        return None
    X_all = df_feat[feat_cols].fillna(0)
    X_val = X_all.iloc[val_idx]
    y_val = y[val_idx]
    X_test = X_all.iloc[test_idx]
    y_test = y[test_idx]

    # probability helper
    try:
        p_val_full = model.predict_proba(X_val)
        p_test_full = model.predict_proba(X_test)
        p_val = p_val_full[:, 1] if p_val_full.shape[1] > 1 else p_val_full.ravel()
        p_test = p_test_full[:, 1] if p_test_full.shape[1] > 1 else p_test_full.ravel()
    except Exception:
        # fallback decision_function
        import numpy as np
        def _to_prob(scores):
            return 1/(1+np.exp(-scores))
        p_val = _to_prob(model.decision_function(X_val))
        p_test = _to_prob(model.decision_function(X_test))

    val_auc = roc_auc_score(y_val, p_val) if len(set(y_val)) > 1 else float('nan')
    test_auc = roc_auc_score(y_test, p_test) if len(set(y_test)) > 1 else float('nan')
    cm = confusion_matrix(y_test, (p_test >= 0.5).astype(int), labels=[0,1])
    return {
        'val_auc': val_auc,
        'test_auc': test_auc,
        'confusion_matrix': cm,
        'feature_columns': feat_cols,
        'test_size': len(test_idx)
    }


def page_model_metrics():
    st.title("Model Metrics")
    metrics = _compute_model_metrics()
    if metrics is None:
        st.info("Unable to compute metrics (missing data/model/features).")
        return
    st.markdown(f"**Validation AUC:** {metrics['val_auc']:.4f}")
    st.markdown(f"**Test AUC:** {metrics['test_auc']:.4f}")
    cm = metrics['confusion_matrix']
    st.subheader("Confusion Matrix (Test)")
    fig, ax = plt.subplots(figsize=(4,3))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred 0','Pred 1'])
    ax.set_yticklabels(['True 0','True 1'])
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center', color='black', fontsize=12)
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    st.caption(f"Test size: {metrics['test_size']} | Features used: {', '.join(metrics['feature_columns'])}")


if __name__ == "__main__":
    main()
