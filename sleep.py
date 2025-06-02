import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model dan scaler
model = joblib.load('sleep.joblib')
scaler = joblib.load('scaler.joblib')

# Label mapping (disesuaikan dengan encoding saat training)
sleep_disorder_labels = {
    0: 'Tidak Ada Gangguan Tidur',
    1: 'Insomnia',
    2: 'Sleep Apnea'
}

st.set_page_config(page_title="Prediksi Gangguan Tidur", layout="centered")
st.title("🛌 Prediksi Gangguan Tidur dengan Machine Learning")
st.subheader("Masukkan detail gaya hidup dan kondisi kesehatan Anda:")

with st.form("input_form"):
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    age = st.number_input("Usia (tahun)", min_value=0, max_value=100, value=25, step=1)
    sleep_duration = st.number_input("Durasi Tidur (jam per hari)", min_value=0.0, max_value=24.0, value=6.0, step=0.1)
    quality = st.number_input("Kualitas Tidur (skala 1-10)", min_value=1, max_value=10, value=5, step=1)
    activity = st.number_input(
        "Aktivitas Fisik Harian (menit/hari)",
        min_value=0, max_value=1440, value=45, step=1,
        help="Berapa menit Anda melakukan aktivitas fisik seperti olahraga setiap hari?"
    )
    stress = st.number_input("Tingkat Stres (skala 1-10)", min_value=0, max_value=10, value=6, step=1)
    bmi_category = st.selectbox("Kategori BMI", ["Normal Weight", "Overweight", "Obese"])
    systolic = st.number_input("Tekanan Darah Sistolik", min_value=70, max_value=200, value=130, step=1)
    diastolic = st.number_input("Tekanan Darah Diastolik", min_value=40, max_value=130, value=85, step=1)
    heart_rate = st.number_input("Detak Jantung (bpm)", min_value=30, max_value=200, value=75, step=1)
    steps = st.number_input("Langkah Harian", min_value=0, max_value=30000, value=5000, step=100)

    submitted = st.form_submit_button("Prediksi")

# Mapping manual sesuai training
gender_encoded = 1 if gender == "Laki-laki" else 0
bmi_map = {"Normal Weight": 0, "Overweight": 1, "Obese": 2}
bmi_encoded = bmi_map[bmi_category]

if submitted:
    # Membuat DataFrame untuk input
    input_array = np.array([[gender_encoded, age, sleep_duration, quality, activity,
                             stress, bmi_encoded, systolic, diastolic, heart_rate, steps]])
    # Normalisasi input
    input_scaled = scaler.transform(input_array)

    # Prediksi
    prediction = model.predict(input_scaled)[0]
    label = sleep_disorder_labels.get(prediction, "Tidak Diketahui")

    # Tampilkan hasil prediksi
    st.success(f"✅ Hasil Prediksi: **{label}**")

    # Saran berdasarkan hasil
    st.markdown("### 💡 Saran:")
    if prediction == 0:
        st.info("Tidak terdeteksi gangguan tidur. Tetap jaga pola hidup sehat dan tidur cukup.")
    elif prediction == 1:
        st.warning("Anda kemungkinan mengalami **Insomnia**.\n\n👉 Kurangi konsumsi kafein, atur jadwal tidur teratur, dan hindari layar sebelum tidur.")
    elif prediction == 2:
        st.warning("Anda kemungkinan mengalami **Sleep Apnea**.\n\n👉 Konsultasikan ke dokter. Hindari alkohol, rokok, dan jaga berat badan ideal.")

    st.markdown("---")
