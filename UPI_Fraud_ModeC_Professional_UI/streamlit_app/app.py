
import streamlit as st
from pathlib import Path
import pandas as pd, joblib, json, numpy as np
import matplotlib.pyplot as plt
st.set_page_config(layout="wide", page_title="UPI Fraud — Mode C (Pro UI)")

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
SRC_DIR = ROOT / "src"
import sys as _sys
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))

# Load artifacts
@st.cache_data(show_spinner=False)
def load_artifacts():
    artifacts = {}
    try:
        artifacts['final_calibrated'] = joblib.load(MODELS_DIR / "final_calibrated_meta.joblib")
    except Exception:
        artifacts['final_calibrated'] = None
    try:
        artifacts['xgb'] = joblib.load(MODELS_DIR / "xgb_raw.joblib")
    except Exception:
        artifacts['xgb'] = None
    try:
        artifacts['iso'] = joblib.load(MODELS_DIR / "isolation_forest.joblib")
    except Exception:
        artifacts['iso'] = None
    try:
        artifacts['meta'] = joblib.load(MODELS_DIR / "meta_lr.joblib")
    except Exception:
        artifacts['meta'] = None
    try:
        artifacts['profiles'] = pd.read_csv(DATA_DIR / "user_profiles_modec.csv")
    except Exception:
        artifacts['profiles'] = pd.DataFrame()
    try:
        artifacts['df'] = pd.read_csv(DATA_DIR / "enhanced_fraud_features_modec.csv", parse_dates=['timestamp'])
    except Exception:
        artifacts['df'] = pd.DataFrame()
    try:
        artifacts['user_thresholds'] = pd.read_csv(DATA_DIR / "user_thresholds_v3.csv")
    except Exception:
        artifacts['user_thresholds'] = pd.DataFrame()
    try:
        artifacts['features'] = pd.read_csv(MODELS_DIR / "feature_list.csv")['feature'].tolist()
    except Exception:
        artifacts['features'] = []
    return artifacts

art = load_artifacts()
st.title("UPI Fraud Detection — Mode C (Professional UI)")
st.markdown("XGBoost + IsolationForest ensemble, calibrated meta-learner, per-user thresholds.")

# Sidebar: select user and quick actions
st.sidebar.header("Controls")
user_list = art['profiles']['user_id'].astype(str).tolist() if not art['profiles'].empty else ["u_0"]
selected_user = st.sidebar.selectbox("Select user", user_list)
amount = st.sidebar.number_input("Amount (₹)", value=100.0, step=10.0)
hour = st.sidebar.slider("Hour of day", 0, 23, 12)
receiver_id = st.sidebar.text_input("Receiver ID", value="r_1")
device_id = st.sidebar.text_input("Device ID", value="dev_100")
txn_type = st.sidebar.selectbox("Transaction type", ["P2P","P2M","AutoPay","QR_Payment"])
sender_loc = st.sidebar.text_input("Sender location", value="Mumbai")
receiver_loc = st.sidebar.text_input("Receiver location", value="Mumbai")

st.sidebar.markdown("---")
if st.sidebar.button("Recompute thresholds (dev)"):
    st.sidebar.info("Recompute feature not enabled in demo. Run train script to rebuild thresholds.")

# Main layout: left - user profile & history; right - prediction & explanations
col1, col2 = st.columns([1.1, 1.4])

with col1:
    st.subheader("User Profile")
    if not art['profiles'].empty:
        prof = art['profiles'][art['profiles']['user_id'].astype(str) == str(selected_user)]
        if not prof.empty:
            st.write(prof.iloc[0].to_dict())
            # show recent transactions for user
            if not art['df'].empty:
                user_tx = art['df'][art['df']['user_id'].astype(str) == str(selected_user)].sort_values('timestamp', ascending=False).head(10)
                st.markdown("**Recent transactions**")
                st.dataframe(user_tx[['timestamp','amount','receiver_id','device_id','transaction_type','is_fraud']].fillna(""), use_container_width=True)
        else:
            st.info("No profile data for selected user.")
    else:
        st.info("User profiles not found in data directory.")

    st.markdown("---")
    st.subheader("Model Diagnostics (Summary)")
    try:
        metrics = json.load(open(MODELS_DIR / "train_metrics.json"))
        st.metric("Test AUC", metrics.get("test_auc", "n/a"))
        st.metric("PR-AUC", metrics.get("pr_auc", "n/a"))
        st.write(metrics)
    except Exception:
        st.info("Train metrics not available.")

    # feature importance chart
    try:
        fi = pd.read_csv(MODELS_DIR / "feature_importances.csv")
        st.markdown("**Top features**")
        st.bar_chart(fi.set_index('feature')['importance'].head(15))
    except Exception:
        pass

with col2:
    st.subheader("Make Prediction")
    if st.button("Predict Transaction"):
        tx = {
            "user_id": selected_user,
            "amount": float(amount),
            "hour": int(hour),
            "receiver_id": receiver_id,
            "device_id": device_id,
            "transaction_type": txn_type,
            "sender_location": sender_loc,
            "receiver_location": receiver_loc
        }
        # call predict module
        from src.predict_modec import predict_single
        prob, decision, details = predict_single(tx, ROOT)
        st.success(f"Decision: {decision} — Fraud probability: {prob:.4f}")
        st.json(details)

        # show feature breakdown (estimated contributions)
        try:
            if art['xgb'] is not None:
                import shap
                # build single-row features
                profiles = art['profiles']
                X = pd.DataFrame([{}])
                # reuse predict_modec's builder logic by importing it (if available)
                from src.predict_modec import build_input
                X = build_input(tx, profiles)
                explainer = shap.TreeExplainer(art['xgb'])
                shap_values = explainer.shap_values(X)
                st.subheader("SHAP Explanation (approx.)")
                # waterfall plot
                fig = shap.plots._waterfall.waterfall_legacy(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X.values[0], feature_names=X.columns), show=False)
                st.pyplot(fig)
        except Exception as e:
            st.info("SHAP explanation not available: " + str(e))

        # Show top contributing features numeric
        st.subheader("Feature values used")
        try:
            from src.predict_modec import build_input
            profiles = art['profiles']
            X = build_input(tx, profiles)
            st.dataframe(X.T, use_container_width=True)
        except Exception as e:
            st.write("Cannot load feature builder:", e)

# Bottom: download artifacts
st.markdown('---')
st.subheader("Download artifacts")
col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    if (DATA_DIR / "enhanced_fraud_features_modec.csv").exists():
        p = DATA_DIR / "enhanced_fraud_features_modec.csv"
        st.download_button("Download dataset CSV", data=p.read_bytes(), file_name=p.name, mime="text/csv")
with col_dl2:
    if (MODELS_DIR / "final_calibrated_meta.joblib").exists():
        m = MODELS_DIR / "final_calibrated_meta.joblib"
        st.download_button("Download model (calibrated)", data=m.read_bytes(), file_name=m.name)
st.caption("Note: This demo UI is for evaluation. For production, secure model access, logging, and auth are required.")
