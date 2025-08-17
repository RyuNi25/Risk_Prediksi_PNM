import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import json

# =========================
# Load model & scaler
# =========================
try:
    model = joblib.load("random_forest_model.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_names = json.load(open("feature_names.json"))
    model_ready = True
except Exception as e:
    st.warning("⚠ Model atau Scaler belum ditemukan. Fitur prediksi mungkin tidak aktif.")
    model, scaler, feature_names = None, None, None
    model_ready = False

# =========================
# Judul
# =========================
st.set_page_config(page_title="Prediksi Risiko Kredit Mekaar PNM", layout="wide")
st.title("🔍 Deteksi Dini Risiko Kredit Mekaar PNM")
st.markdown("Aplikasi interaktif untuk **analisis data pinjaman** dan **prediksi risiko kredit** nasabah berdasarkan data historis.")

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["🧑‍💻 Prediksi Input Manual", "📂 Prediksi dari Dataset CSV"])

# =========================
# Tab 1: Input Manual
# =========================
with tab1:
    st.sidebar.header("Masukkan Data Nasabah")

    ODInterest = st.sidebar.number_input("ODInterest (Tunggakan Bunga)", min_value=0.0)
    ODPrincipal = st.sidebar.number_input("ODPrincipal (Tunggakan Pokok)", min_value=0.0)
    PrincipalDue = st.sidebar.number_input("PrincipalDue (Pokok Terhutang)", min_value=0.0)
    InterestDue = st.sidebar.number_input("InterestDue (Bunga Terhutang)", min_value=0.0)
    NoOfArrearDays = st.sidebar.number_input("NoOfArrearDays (Hari Tunggakan)", min_value=0)

    if st.sidebar.button("Prediksi Risiko"):
        if model_ready:
            # Susun input
            input_data = pd.DataFrame([[
                ODInterest, ODPrincipal, PrincipalDue, InterestDue, NoOfArrearDays
            ]], columns=["ODInterest", "ODPrincipal", "PrincipalDue", "InterestDue", "NoOfArrearDays"])

            # Sesuaikan dengan feature_names
            input_data = input_data.reindex(columns=feature_names, fill_value=0)

            # Scaling
            try:
                input_scaled = scaler.transform(input_data)
            except Exception as e:
                st.warning(f"⚠ Gagal scaling: {e}")
                input_scaled = input_data

            # Prediksi
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0][1]

            # Hasil
            st.subheader("📊 Hasil Prediksi")
            if prediction == 1:
                st.error(f"⚠ Risiko Tinggi — Probabilitas: {proba:.2%}")
            else:
                st.success(f"✅ Risiko Rendah — Probabilitas: {proba:.2%}")

            # Feature Importance
            if hasattr(model, "feature_importances_"):
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
        else:
            st.error("❌ Model atau Scaler belum tersedia. Upload file terlebih dahulu.")

# =========================
# Tab 2: Prediksi dari CSV
# =========================
with tab2:
    st.subheader("📂 Prediksi Risiko dari Dataset CSV")
    uploaded_csv = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_csv is not None:
        if model_ready:
            df = pd.read_csv(uploaded_csv)
            st.write("📋 Data yang diupload:", df.head())

            # Reindex agar sesuai feature training
            df_input = df.reindex(columns=feature_names, fill_value=0)

            # Scaling
            try:
                X_scaled = scaler.transform(df_input)
            except Exception as e:
                st.warning(f"⚠ Gagal scaling: {e}")
                X_scaled = df_input

            # Prediksi
            df["Prediksi"] = model.predict(X_scaled)
            df["Prob_RisikoTinggi"] = model.predict_proba(X_scaled)[:, 1]

            # Tampilkan hasil
            st.subheader("📊 Hasil Prediksi Dataset")
            st.write(df[["Prediksi", "Prob_RisikoTinggi"]].head(20))

            # Download hasil prediksi
            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Download Hasil Prediksi", data=csv_out, file_name="prediksi_risiko.csv", mime="text/csv")
        else:
            st.error("❌ Model & Scaler belum tersedia. Silakan upload file model terlebih dahulu.")
