import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import json
from io import BytesIO

st.set_page_config(page_title="Prediksi Risiko Kredit Mekaar PNM", layout="wide")
st.title("🔍 Deteksi Dini Risiko Kredit Mekaar PNM")
st.markdown("Aplikasi interaktif untuk **analisis data pinjaman** dan **prediksi risiko kredit** nasabah berdasarkan data historis.")

# =========================
# Upload Section
# =========================
st.sidebar.header("📂 Upload File")
uploaded_csv = st.sidebar.file_uploader("Upload Dataset (.csv)", type=["csv"])
uploaded_model = st.sidebar.file_uploader("Upload Model (.joblib)", type=["joblib"])
uploaded_scaler = st.sidebar.file_uploader("Upload Scaler (.joblib)", type=["joblib"])
uploaded_features = st.sidebar.file_uploader("Upload Feature Names (.json)", type=["json"])

# =========================
# Load Model & Scaler
# =========================
model, scaler, feature_names = None, None, None

if uploaded_model is not None:
    model = joblib.load(uploaded_model)

if uploaded_scaler is not None:
    scaler = joblib.load(uploaded_scaler)

if uploaded_features is not None:
    feature_names = json.load(uploaded_features)

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["📊 Analisis Data Pinjaman", "🤖 Prediksi Risiko Kredit"])

# =============================
# Tab 1 - Analisis Data Pinjaman
# =============================
with tab1:
    st.header("📊 Analisis Data Pinjaman Nasabah")

    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(uploaded_csv, delimiter=";")

        st.subheader("📑 Data Nasabah")
        st.dataframe(df.head())

        st.subheader("📊 Ringkasan Data")
        st.write(df.describe(include="all"))

        st.subheader("📌 Korelasi Antar Variabel")
        try:
            corr = df.corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            cax = ax.matshow(corr, cmap="coolwarm")
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
            plt.yticks(range(len(corr.columns)), corr.columns)
            fig.colorbar(cax)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Korelasi tidak dapat dihitung: {e}")
    else:
        st.info("Silakan upload dataset .csv terlebih dahulu.")

# ==============================
# Tab 2 - Prediksi Risiko Kredit
# ==============================
with tab2:
    st.header("🤖 Prediksi Risiko Kredit")

    if model is None or scaler is None:
        st.error("⚠ Model & Scaler belum tersedia. Silakan upload file model dan scaler terlebih dahulu.")
    else:
        if feature_names is None:
            st.warning("⚠ Feature names JSON tidak ditemukan. Input field mungkin tidak sesuai.")
            feature_names = ["ODInterest", "ODPrincipal", "PrincipalDue", "InterestDue", "NoOfArrearDays"]

        st.subheader("📥 Masukkan Data Nasabah")
        input_data = {}
        for feat in feature_names:
            if "Day" in feat or "Term" in feat:
                val = st.number_input(feat, min_value=0, value=0)
            else:
                val = st.number_input(feat, min_value=0.0, value=0.0)
            input_data[feat] = val

        if st.button("🔮 Prediksi Risiko"):
            df_input = pd.DataFrame([input_data])

            try:
                input_scaled = scaler.transform(df_input)
            except Exception:
                input_scaled = df_input

            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0][1]

            st.subheader("📊 Hasil Prediksi")
            if prediction == 1:
                st.error(f"⚠ Risiko Tinggi — Probabilitas: {proba:.2%}")
            else:
                st.success(f"✅ Risiko Rendah — Probabilitas: {proba:.2%}")

            # =====================
            # Export hasil prediksi
            # =====================
            df_input["Prediksi"] = ["Risiko Tinggi" if prediction == 1 else "Risiko Rendah"]
            df_input["Probabilitas Risiko Tinggi"] = proba

            st.subheader("💾 Simpan Hasil Prediksi")
            csv = df_input.to_csv(index=False).encode("utf-8")
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                df_input.to_excel(writer, index=False, sheet_name="Hasil Prediksi")

            st.download_button("⬇️ Download CSV", data=csv, file_name="hasil_prediksi.csv", mime="text/csv")
            st.download_button("⬇️ Download Excel", data=excel_buffer.getvalue(),
                               file_name="hasil_prediksi.xlsx", mime="application/vnd.ms-excel")

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
