import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# Konfigurasi Halaman
# =========================
st.set_page_config(page_title="Prediksi Risiko Kredit Mekaar PNM", layout="wide")
st.title("🔍 Deteksi Dini Risiko Kredit Mekaar PNM")
st.markdown("Aplikasi interaktif untuk memprediksi risiko kredit nasabah berdasarkan data historis.")

# =========================
# Load Model & Scaler Aman
# =========================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("random_forest_model.joblib")
    except Exception as e:
        st.stop()
        st.error(f"❌ Gagal memuat model: {e}")

    try:
        scaler = joblib.load("scaler.joblib")
    except FileNotFoundError:
        scaler = None
        st.warning("⚠ Scaler tidak ditemukan. Data akan digunakan tanpa scaling.")
    return model, scaler

model, scaler = load_assets()

# =========================
# Form Input
# =========================
st.sidebar.header("Masukkan Data Nasabah")
input_fields = {
    "ODInterest": st.sidebar.number_input("ODInterest (Tunggakan Bunga)", min_value=0.0),
    "ODPrincipal": st.sidebar.number_input("ODPrincipal (Tunggakan Pokok)", min_value=0.0),
    "PrincipalDue": st.sidebar.number_input("PrincipalDue (Pokok Terhutang)", min_value=0.0),
    "InterestDue": st.sidebar.number_input("InterestDue (Bunga Terhutang)", min_value=0.0),
    "NoOfArrearDays": st.sidebar.number_input("NoOfArrearDays (Hari Tunggakan)", min_value=0)
}

# =========================
# Prediksi
# =========================
if st.sidebar.button("Prediksi Risiko"):
    input_df = pd.DataFrame([input_fields])

    # Scaling jika scaler ada
    if scaler:
        try:
            input_scaled = scaler.transform(input_df)
        except Exception as e:
            st.error(f"❌ Gagal melakukan scaling: {e}")
            st.stop()
    else:
        input_scaled = input_df

    # Prediksi
    try:
        prediction = model.predict(input_scaled)[0]
    except Exception as e:
        st.error(f"❌ Gagal melakukan prediksi: {e}")
        st.stop()

    # Probabilitas (jika ada)
    try:
        proba = model.predict_proba(input_scaled)[0][1]
    except AttributeError:
        proba = None

    # Hasil
    st.subheader("📊 Hasil Prediksi")
    if prediction == 1:
        st.error(f"⚠ Risiko Tinggi" + (f" — Probabilitas: {proba:.2%}" if proba is not None else ""))
    else:
        st.success(f"✅ Risiko Rendah" + (f" — Probabilitas: {proba:.2%}" if proba is not None else ""))

# =========================
# Feature Importance
# =========================
if hasattr(model, "feature_importances_"):
    st.subheader("📌 Faktor Terpenting dalam Prediksi")
    feature_importance = pd.DataFrame({
        "Fitur": list(input_fields.keys()),
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(feature_importance["Fitur"], feature_importance["Importance"], color="skyblue")
    ax.set_xlabel("Tingkat Kepentingan")
    ax.set_ylabel("Fitur")
    st.pyplot(fig)
