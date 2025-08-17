import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Deteksi Risiko Kredit Mekaar PNM", layout="wide")

st.title("🔍 Deteksi Dini Risiko Kredit Mekaar PNM")
st.write("Aplikasi interaktif untuk analisis data pinjaman dan prediksi risiko kredit nasabah.")


# ================================
# 1. Upload Data Nasabah
# ================================
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Nasabah")
    st.dataframe(data.head())

    # Statistik deskriptif
    st.subheader("📈 Statistik Deskriptif")
    st.write(data.describe())

    # Korelasi antar variabel
    st.subheader("🔗 Korelasi Antar Variabel")
    corr = data.corr(numeric_only=True)
    fig, ax = plt.subplots()
    cax = ax.matshow(corr, cmap="coolwarm")
    fig.colorbar(cax)
    st.pyplot(fig)


# ================================
# 2. Upload Model & Scaler
# ================================
st.sidebar.header("🤖 Upload Model")
uploaded_model = st.sidebar.file_uploader("Upload file model (joblib)", type=["joblib"])
uploaded_scaler = st.sidebar.file_uploader("Upload file scaler (joblib)", type=["joblib"])

model = None
scaler = None

if uploaded_model is not None:
    model = joblib.load(uploaded_model)
if uploaded_scaler is not None:
    scaler = joblib.load(uploaded_scaler)

if model is not None and scaler is not None:
    st.sidebar.success("✅ Model & Scaler berhasil dimuat.")
else:
    st.sidebar.warning("⚠️ Model & Scaler belum tersedia. Upload dulu file `.joblib`.")


# ================================
# 3. Prediksi Risiko Kredit
# ================================
st.subheader("🧮 Prediksi Risiko Kredit")

if model is not None and scaler is not None and uploaded_file:
    # Pilih 1 nasabah untuk prediksi
    nasabah_index = st.number_input("Pilih index nasabah untuk prediksi", min_value=0, max_value=len(data)-1, value=0)
    nasabah_data = data.iloc[nasabah_index:nasabah_index+1]

    st.write("📌 Data Nasabah Terpilih")
    st.write(nasabah_data)

    # Preprocessing
    try:
        X_scaled = scaler.transform(nasabah_data.select_dtypes(include=np.number))

        # Prediksi
        prediction = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0][1]  # probabilitas risiko

        st.subheader("📢 Hasil Prediksi")
        if prediction == 1:
            st.error(f"🚨 Nasabah ini berisiko Gagal Bayar. Probabilitas: {proba:.2f}")
        else:
            st.success(f"✅ Nasabah ini diprediksi Aman Bayar. Probabilitas Risiko: {proba:.2f}")
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
else:
    st.info("Upload dataset + model + scaler untuk mulai prediksi.")
