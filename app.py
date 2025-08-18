import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import json
from io import StringIO, BytesIO   # <- pakai dua-duanya

# =========================
# Konfigurasi Halaman
# =========================
st.set_page_config(page_title="Prediksi Risiko Kredit Mekaar PNM", layout="wide")
st.title("🔍 Deteksi Dini Risiko Kredit Mekaar PNM")
st.write("Aplikasi interaktif untuk **analisis data pinjaman** dan **prediksi risiko kredit** nasabah berdasarkan data historis.")

# =========================
# Upload Dataset
# =========================
st.sidebar.header("📂 Upload Dataset")
uploaded_csv = st.sidebar.file_uploader("Upload file CSV", type=["csv"])

df = None
if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv, sep=None, engine="python")
        st.subheader("📊 Data Nasabah")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Gagal membaca file CSV: {e}")
        st.stop()

    # Ringkasan Data
    st.subheader("📋 Ringkasan Data")
    st.write(df.describe(include="all"))

    # Korelasi Antar Variabel
    st.subheader("📌 Korelasi Antar Variabel")
    try:
        corr = df.corr(numeric_only=True)
        st.write(corr)
        fig, ax = plt.subplots()
        cax = ax.matshow(corr, cmap="coolwarm")
        plt.colorbar(cax)
        st.pyplot(fig)
    except Exception as e:
        st.warning("Tidak bisa menghitung korelasi: " + str(e))

# =========================
# Upload Model
# =========================
st.sidebar.header("🤖 Upload Model")
uploaded_model = st.sidebar.file_uploader("Upload file model (.joblib)", type=["joblib"])
uploaded_scaler = st.sidebar.file_uploader("Upload file scaler (.joblib)", type=["joblib"])
uploaded_features = st.sidebar.file_uploader("Upload feature names (.json)", type=["json"])

model, scaler, feature_names = None, None, None

if uploaded_model is not None:
    model = joblib.load(uploaded_model)

if uploaded_scaler is not None:
    scaler = joblib.load(uploaded_scaler)

if uploaded_features is not None:
    feature_names = json.load(uploaded_features)

# Default feature names jika JSON tidak ada
if feature_names is None:
    st.warning("⚠️ Feature names JSON tidak ditemukan. Input field mungkin tidak sesuai.")
    feature_names = ["ODInterest", "ODPrincipal", "PrincipalDue", "InterestDue", "NoOfArrearDays"]

# =========================
# Prediksi Risiko Kredit
# =========================
st.subheader("🤝 Prediksi Risiko Kredit")

if model is None or scaler is None:
    st.error("⚠️ Model & Scaler belum tersedia. Silakan upload file model dan scaler terlebih dahulu.")
else:
    st.subheader("📝 Masukkan Data Nasabah")
    input_data = {}
    for feat in feature_names:
        if "Day" in feat or "Term" in feat:   # kemungkinan integer
            val = st.number_input(feat, min_value=0, value=0)
        else:  # kemungkinan float
            val = st.number_input(feat, min_value=0.0, value=0.0)
        input_data[feat] = val

    if st.button("🔮 Prediksi Risiko"):
        try:
            input_df = pd.DataFrame([input_data])
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]
            proba = model.predict_proba(scaled_input)[0][1]

            if prediction == 1:
                st.error(f"⚠️ Risiko Tinggi – Probabilitas: {proba:.2%}")
            else:
                st.success(f"✅ Risiko Rendah – Probabilitas: {proba:.2%}")

            # =========================
            # Export Hasil Prediksi
            # =========================
            df_input = input_df.copy()
            df_input["Prediksi"] = ["Risiko Tinggi" if prediction == 1 else "Risiko Rendah"]
            df_input["Probabilitas Risiko Tinggi"] = proba

            st.subheader("💾 Simpan Hasil Prediksi")
            csv = df_input.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", data=csv, file_name="hasil_prediksi.csv", mime="text/csv")

            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                df_input.to_excel(writer, index=False, sheet_name="HasilPrediksi")
            st.download_button(
                "⬇️ Download Excel",
                data=excel_buffer.getvalue(),
                file_name="hasil_prediksi.xlsx",
                mime="application/vnd.ms-excel"
            )

            # =========================
            # Feature Importance
            # =========================
            if hasattr(model, "feature_importances_"):
                st.subheader("📊 Faktor Terpenting dalam Prediksi")
                feature_importance = pd.DataFrame({
                    "Fitur": feature_names,
                    "Importance": model.feature_importances_
                }).sort_values(by="Importance", ascending=False)

                st.bar_chart(feature_importance.set_index("Fitur"))
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")
