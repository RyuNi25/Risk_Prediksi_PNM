import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# =====================
# Load Model & Scaler
# =====================
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# =====================
# Judul Aplikasi
# =====================
st.title("📊 Sistem Evaluasi Kelayakan Kredit Mikro")
st.markdown("Aplikasi ini memprediksi kelayakan kredit berdasarkan data historis nasabah.")

# =====================
# Input Data
# =====================
st.subheader("Masukkan Data Nasabah")

feature_names = model.feature_names_in_

# Form input untuk setiap fitur
input_data = []
for feature in feature_names:
    val = st.number_input(f"{feature}", value=0.0)
    input_data.append(val)

# =====================
# Prediksi
# =====================
if st.button("Prediksi"):
    # Ubah menjadi array 2D
    input_array = np.array(input_data).reshape(1, -1)

    # Scaling
    input_scaled = scaler.transform(input_array)

    # Prediksi
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    st.write(f"### Hasil Prediksi: **{'Layak' if pred == 1 else 'Tidak Layak'}**")
    st.write(f"Probabilitas Layak: {proba[1]*100:.2f}%")
    st.write(f"Probabilitas Tidak Layak: {proba[0]*100:.2f}%")

    # =====================
    # Feature Importance
    # =====================
    st.subheader("📌 Feature Importance")
    feature_importance = pd.DataFrame({
        "Fitur": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(feature_importance)

    # Plot Grafik
    fig, ax = plt.subplots()
    ax.barh(feature_importance["Fitur"], feature_importance["Importance"], color='skyblue')
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_ylabel("Fitur")
    ax.set_title("Feature Importance")
    st.pyplot(fig)
