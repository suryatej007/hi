import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import joblib
from tensorflow.keras.models import load_model

# === Load Data & Models ===
@st.cache_data
def load_all():
    df = pd.read_csv("NSL_KDD_5features_from_github.csv")
    scaler = joblib.load("scaler.pkl")
    rf_model = joblib.load("random_forest.pkl")
    autoencoder = load_model("autoencoder.h5", compile=False)
    return df, scaler, rf_model, autoencoder

df, scaler, rf_model, autoencoder = load_all()

# === Preprocess ===
X = df.drop("label", axis=1)
y = df["label"]
X_scaled = scaler.transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# === Threshold from Autoencoder ===
recon_val = autoencoder.predict(X_val)
recon_error_val = np.mean(np.square(X_val - recon_val), axis=1)
threshold = np.percentile(recon_error_val[y_val == 0], 99)

# === Predictions ===
reconstructed = autoencoder.predict(X_scaled)
recon_error = np.mean(np.square(X_scaled - reconstructed), axis=1)
ae_pred = (recon_error > threshold).astype(int)
ae_score = np.clip(recon_error / threshold, 0, 1)

rf_pred = rf_model.predict(X_scaled)
rf_prob = rf_model.predict_proba(X_scaled)[:, 1]

alpha = 0.7
combined_prob = alpha * rf_prob + (1 - alpha) * ae_score
combined_pred = (combined_prob > 0.5).astype(int)

# === Streamlit UI ===
st.title("Cloud Intrusion Detection: RF + Autoencoder")

# === Tabs ===
tabs = st.tabs(["📊 Evaluation", "🔍 Single Prediction"])

# === Evaluation Tab ===
with tabs[0]:
    st.subheader("Classification Reports")
    st.text("Random Forest:\n" + classification_report(y, rf_pred))
    st.text("Autoencoder:\n" + classification_report(y, ae_pred))
    st.text("Combined Model:\n" + classification_report(y, combined_pred))

    def plot_conf_matrix(cm, title, cmap):
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(title)
        st.pyplot(fig)

    st.subheader("Confusion Matrices")
    plot_conf_matrix(confusion_matrix(y, rf_pred), "Random Forest", "Greens")
    plot_conf_matrix(confusion_matrix(y, ae_pred), "Autoencoder", "Oranges")
    plot_conf_matrix(confusion_matrix(y, combined_pred), "Combined Model", "Blues")

    st.subheader("ROC Curve Comparison")
    fpr_rf, tpr_rf, _ = roc_curve(y, rf_prob)
    fpr_ae, tpr_ae, _ = roc_curve(y, ae_score)
    fpr_comb, tpr_comb, _ = roc_curve(y, combined_prob)

    auc_rf = auc(fpr_rf, tpr_rf)
    auc_ae = auc(fpr_ae, tpr_ae)
    auc_comb = auc(fpr_comb, tpr_comb)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr_rf, tpr_rf, label=f"RF (AUC = {auc_rf:.2f})", lw=2)
    ax.plot(fpr_ae, tpr_ae, label=f"AE (AUC = {auc_ae:.2f})", lw=2)
    ax.plot(fpr_comb, tpr_comb, label=f"Combined (AUC = {auc_comb:.2f})", lw=2, color="darkorange")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Autoencoder Reconstruction Error Distribution")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(recon_error[y == 0], bins=100, color='green', label="Normal", stat="density", ax=ax)
    sns.histplot(recon_error[y == 1], bins=100, color='red', label="Attack", stat="density", ax=ax)
    ax.axvline(threshold, color='black', linestyle='--', label=f"Threshold = {threshold:.4f}")
    ax.set_title("Reconstruction Error Distribution")
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# === Single Prediction Tab ===
with tabs[1]:
    st.subheader("Predict a Single Instance")

    duration = st.number_input("Duration", min_value=0.0)
    src_bytes = st.number_input("Src Bytes", min_value=0.0)
    dst_bytes = st.number_input("Dst Bytes", min_value=0.0)
    count = st.number_input("Count", min_value=0.0)
    serror_rate = st.number_input("Serror Rate", min_value=0.0, max_value=1.0)

    input_array = np.array([[duration, src_bytes, dst_bytes, count, serror_rate]])

    if st.button("Predict"):
        scaled_input = scaler.transform(input_array)

        # AE
        recon = autoencoder.predict(scaled_input)
        error = np.mean(np.square(scaled_input - recon), axis=1)
        ae_prediction = (error > threshold).astype(int)
        ae_score_single = np.clip(error / threshold, 0, 1)

        # RF
        rf_prediction = rf_model.predict(scaled_input)[0]
        rf_probability = rf_model.predict_proba(scaled_input)[:, 1][0]

        # Combined
        combined_score = alpha * rf_probability + (1 - alpha) * ae_score_single
        combined_prediction = (combined_score > 0.5).astype(int)

        st.markdown("### 🔍 Prediction Results")
        st.write(f"**Random Forest**: {'🛑 Attack' if rf_prediction == 1 else '✅ Normal'} (Prob: {rf_probability:.4f})")
        st.write(f"**Autoencoder**: {'🛑 Attack' if ae_prediction[0] == 1 else '✅ Normal'} (Reconstruction Error: {error[0]:.6f})")
        st.write(f"**Combined Model**: {'🛑 Attack' if combined_prediction[0] == 1 else '✅ Normal'} (Score: {combined_score[0]:.4f})")
