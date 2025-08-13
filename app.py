import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Load model & data scaler
# =========================
model = joblib.load("random_forest_model.pkl")  # Simpan model ini dari Colab kamu
scaler = joblib.load("scaler.pkl")              # Kalau pakai StandardScaler di preprocessing

# =========================
# Judul
# =========================
st.set_page_config(page_title="Prediksi Risiko Kredit Mekaar PNM", layout="wide")
st.title("🔍 Deteksi Dini Risiko Kredit Mekaar PNM")
st.markdown("Aplikasi interaktif untuk memprediksi risiko kredit nasabah berdasarkan data historis.")

# =========================
# Form Input
# =========================
st.sidebar.header("Masukkan Data Nasabah")

# Contoh field input (sesuaikan dengan fitur model kamu)
ODInterest = st.sidebar.number_input("ODInterest (Tunggakan Bunga)", min_value=0.0)
ODPrincipal = st.sidebar.number_input("ODPrincipal (Tunggakan Pokok)", min_value=0.0)
PrincipalDue = st.sidebar.number_input("PrincipalDue (Pokok Terhutang)", min_value=0.0)
InterestDue = st.sidebar.number_input("InterestDue (Bunga Terhutang)", min_value=0.0)
NoOfArrearDays = st.sidebar.number_input("NoOfArrearDays (Hari Tunggakan)", min_value=0)

# Jika ada fitur tambahan, tambahkan di sini...

# =========================
# Prediksi
# =========================
if st.sidebar.button("Prediksi Risiko"):
    # Susun input user jadi DataFrame
    input_data = pd.DataFrame([[
        ODInterest, ODPrincipal, PrincipalDue, InterestDue, NoOfArrearDays
    ]], columns=["ODInterest", "ODPrincipal", "PrincipalDue", "InterestDue", "NoOfArrearDays"])

    # Scaling jika diperlukan
    try:
        input_scaled = scaler.transform(input_data)
    except:
        input_scaled = input_data

    # Prediksi
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]  # Probabilitas risiko tinggi

    # Hasil
    st.subheader("📊 Hasil Prediksi")
    if prediction == 1:
        st.error(f"⚠ Risiko Tinggi — Probabilitas: {proba:.2%}")
    else:
        st.success(f"✅ Risiko Rendah — Probabilitas: {proba:.2%}")

# =========================
# Feature Importance
# =========================
if hasattr(model, "feature_importances_"):
    st.subheader("📌 Faktor Terpenting dalam Prediksi")
    feature_importance = pd.DataFrame({
        "Fitur": ["ODInterest", "ODPrincipal", "PrincipalDue", "InterestDue", "NoOfArrearDays"],
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots()
    ax.barh(feature_importance["Fitur"], feature_importance["Importance"], color="skyblue")
    ax.set_xlabel("Tingkat Kepentingan")
    ax.set_ylabel("Fitur")
    ax.invert_yaxis()
    st.pyplot(fig)
