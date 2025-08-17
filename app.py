import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json

st.set_page_config(page_title="Prediksi Risiko Kredit Mekaar PNM", layout="wide")

st.title("🔍 Deteksi Dini Risiko Kredit Mekaar PNM")
st.write("Aplikasi interaktif untuk analisis data pinjaman dan prediksi risiko kredit nasabah.")

# ================================
# 1. Load Model, Scaler, Feature Names
# ================================
try:
    model = joblib.load("random_forest_model.joblib")
    scaler = joblib.load("scaler.joblib")

    with open("feature_names.json", "r") as f:
        feature_names = json.load(f)

    st.sidebar.success("✅ Model & Scaler berhasil dimuat.")
except Exception as e:
    st.sidebar.error(f"⚠️ Gagal load model atau scaler: {e}")
    model, scaler, feature_names = None, None, []

# ================================
# 2. Input Manual Nasabah
# ================================
st.sidebar.header("📌 Masukkan Data Nasabah")

ODInterest = st.sidebar.number_input("ODInterest (Tunggakan Bunga)", min_value=0.0, value=0.0)
ODPrincipal = st.sidebar.number_input("ODPrincipal (Tunggakan Pokok)", min_value=0.0, value=0.0)
PrincipalDue = st.sidebar.number_input("PrincipalDue (Pokok Terhutang)", min_value=0.0, value=0.0)
InterestDue = st.sidebar.number_input("InterestDue (Bunga Terhutang)", min_value=0.0, value=0.0)
NoOfArrearDays = st.sidebar.number_input("NoOfArrearDays (Hari Tunggakan)", min_value=0, value=0)

# ================================
# 3. Prediksi Risiko Kredit
# ================================
if st.sidebar.button("Prediksi Risiko"):

    if model is None or scaler is None or not feature_names:
        st.error("⚠️ Model/Scaler/Feature Names belum dimuat.")
    else:
        # Susun input ke DataFrame
        input_data = pd.DataFrame([[
            ODInterest, ODPrincipal, PrincipalDue, InterestDue, NoOfArrearDays
        ]], columns=["ODInterest", "ODPrincipal", "PrincipalDue", "InterestDue", "NoOfArrearDays"])

        # Reindex agar sesuai dengan urutan fitur saat training
        input_data = input_data.reindex(columns=feature_names, fill_value=0)

        # Scaling
        X_scaled = scaler.transform(input_data)

        # Prediksi
        prediction = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]

        st.subheader("📊 Hasil Prediksi")
        if prediction == 1:
            st.error(f"🚨 Risiko Tinggi — Probabilitas: {proba[1]:.2%}")
        else:
            st.success(f"✅ Risiko Rendah — Probabilitas: {proba[0]:.2%}")

# ================================
# 4. Feature Importance
# ================================
if model is not None and hasattr(model, "feature_importances_"):
    st.subheader("📌 Faktor Terpenting dalam Prediksi")
    feature_importance = pd.DataFrame({
        "Fitur": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots()
    ax.barh(feature_importance["Fitur"], feature_importance["Importance"], color="skyblue")
    ax.set_xlabel("Tingkat Kepentingan")
    ax.set_ylabel("Fitur")
    ax.invert_yaxis()
    st.pyplot(fig)
