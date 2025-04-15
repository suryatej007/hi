import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# === Load Data ===
df = pd.read_csv("NSL_KDD_5features_from_github.csv")
X = df.drop("label", axis=1)
y = df["label"]

# === Load Scaler and Scale Features ===
scaler = joblib.load("scaler.pkl")
X_scaled = scaler.transform(X)

# === Train/Validation Split for Threshold Calculation ===
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# === Load Models ===
rf_model = joblib.load("random_forest.pkl")
autoencoder = load_model("autoencoder.h5", compile=False)

# === Autoencoder Threshold (from normal validation data) ===
recon_val = autoencoder.predict(X_val)
recon_error_val = np.mean(np.square(X_val - recon_val), axis=1)
threshold = np.percentile(recon_error_val[y_val == 0], 99)

# === Streamlit App UI ===
st.title("Cloud Intrusion Detection System")
st.subheader("Enter Network Traffic Features")

duration = st.number_input("Duration", min_value=0.0)
src_bytes = st.number_input("Src Bytes", min_value=0.0)
dst_bytes = st.number_input("Dst Bytes", min_value=0.0)
count = st.number_input("Count", min_value=0.0)
serror_rate = st.number_input("Serror Rate", min_value=0.0)

user_input = np.array([duration, src_bytes, dst_bytes, count, serror_rate]).reshape(1, -1)

if st.button("Run Prediction"):
    scaled_input = scaler.transform(user_input)

    # Autoencoder Prediction
    reconstructed = autoencoder.predict(scaled_input)
    recon_error = np.mean(np.square(scaled_input - reconstructed), axis=1)
    ae_pred = (recon_error > threshold).astype(int)
    ae_score = np.clip(recon_error / threshold, 0, 1)

    # Random Forest Prediction
    rf_pred = rf_model.predict(scaled_input)[0]
    rf_prob = rf_model.predict_proba(scaled_input)[:, 1][0]

    # Combined Model
    alpha = 0.7
    combined_prob = alpha * rf_prob + (1 - alpha) * ae_score
    combined_pred = (combined_prob > 0.5).astype(int)

    result = "🔴 Attack Detected" if combined_pred == 1 else "🟢 Normal Traffic"

    st.markdown(f"### Prediction Result: {result}")
    st.markdown("### Model Insights:")
    st.write(f"**Random Forest Prediction**: {'Attack' if rf_pred == 1 else 'Normal'}")
    st.write(f"**Autoencoder Reconstruction Error**: {recon_error[0]:.6f}")
    st.write(f"**Autoencoder Threshold**: {threshold:.6f}")
    st.write(f"**Autoencoder Prediction**: {'Attack' if ae_pred[0] == 1 else 'Normal'}")
    st.write(f"**Combined Model Prediction**: {'Attack' if combined_pred[0] == 1 else 'Normal'}")
