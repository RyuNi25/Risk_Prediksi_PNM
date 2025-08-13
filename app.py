import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# =========================
# Konfigurasi Halaman
# =========================
st.set_page_config(page_title="Prediksi Risiko Kredit Mekaar PNM", layout="wide")
st.title("🔍 Deteksi Dini Risiko Kredit Mekaar PNM")
st.markdown("Aplikasi interaktif untuk **analisis data pinjaman** dan **prediksi risiko kredit** nasabah berdasarkan data historis.")

# =========================
# Load Model & Scaler (jika ada)
# =========================
model, scaler = None, None
if os.path.exists("random_forest_model.joblib") and os.path.exists("scaler.joblib"):
    model = joblib.load("random_forest_model.joblib")
    scaler = joblib.load("scaler.joblib")
else:
    st.warning("⚠ Model atau scaler belum ditemukan. Fitur prediksi mungkin tidak aktif.")

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["📊 Analisis Data Pinjaman", "🤖 Prediksi Risiko Kredit"])

# =========================
# TAB 1: Analisis Data
# =========================
with tab1:
    st.header("📊 Analisis Data Pinjaman Nasabah")

    uploaded_file = st.file_uploader("Unggah dataset pinjaman (.csv)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("📋 Ringkasan Data")
        st.dataframe(df.head())

        st.subheader("📈 Statistik Deskriptif")
        st.write(df.describe())

        # Grafik distribusi jumlah pinjaman
        if "PrincipalDue" in df.columns:
            st.subheader("📊 Distribusi Pokok Terhutang (PrincipalDue)")
            fig, ax = plt.subplots()
            ax.hist(df["PrincipalDue"], bins=30, color="skyblue", edgecolor="black")
            ax.set_xlabel("PrincipalDue")
            ax.set_ylabel("Jumlah Nasabah")
            st.pyplot(fig)

        # Korelasi antar variabel
        st.subheader("📌 Korelasi Antar Variabel")
        corr = df.corr(numeric_only=True)
        st.dataframe(corr)

# =========================
# TAB 2: Prediksi Risiko Kredit
# =========================
with tab2:
    st.header("🤖 Prediksi Risiko Kredit")

    if model is None or scaler is None:
        st.error("⚠ Model & Scaler belum tersedia. Silakan upload file model terlebih dahulu.")
    else:
        st.sidebar.header("Masukkan Data Nasabah")
        ODInterest = st.sidebar.number_input("ODInterest (Tunggakan Bunga)", min_value=0.0)
        ODPrincipal = st.sidebar.number_input("ODPrincipal (Tunggakan Pokok)", min_value=0.0)
        PrincipalDue = st.sidebar.number_input("PrincipalDue (Pokok Terhutang)", min_value=0.0)
        InterestDue = st.sidebar.number_input("InterestDue (Bunga Terhutang)", min_value=0.0)
        NoOfArrearDays = st.sidebar.number_input("NoOfArrearDays (Hari Tunggakan)", min_value=0)

        if st.sidebar.button("Prediksi Risiko"):
            input_data = pd.DataFrame([[
                ODInterest, ODPrincipal, PrincipalDue, InterestDue, NoOfArrearDays
            ]], columns=["ODInterest", "ODPrincipal", "PrincipalDue", "InterestDue", "NoOfArrearDays"])

            try:
                input_scaled = scaler.transform(input_data)
            except Exception as e:
                st.warning(f"Gagal scaling: {e}")
                input_scaled = input_data

            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0][1]

            st.subheader("📊 Hasil Prediksi")
            if prediction == 1:
                st.error(f"⚠ Risiko Tinggi — Probabilitas: {proba:.2%}")
            else:
                st.success(f"✅ Risiko Rendah — Probabilitas: {proba:.2%}")

            if hasattr(model, "feature_importances_"):
                st.subheader("📌 Faktor Terpenting dalam Prediksi")
                feature_importance = pd.DataFrame({
                    "Fitur": ["ODInterest", "ODPrincipal", "PrincipalDue", "InterestDue", "NoOfArrearDays"],
                    "Importance": model.feature_importances_
                }).sort_values(by="Importance", ascending=False)

                fig, ax = plt.subplots()
                ax.barh(feature_importance["Fitur"], feature_importance["Importance"], color="skyblue")
                ax.set_xlabel("Tingkat Kepentingan")
                ax.invert_yaxis()
                st.pyplot(fig)
