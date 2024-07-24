import sklearn
import pickle
import streamlit as st
import numpy as np

# Membaca Model dan Skaler
with open('maternal_risk_model.sav', 'rb') as file:
    maternalrisk_model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Judul Web
st.title('Prediksi Risiko pada Ibu')

# Membagi Tampilan Kolom
col1, col2 = st.columns(2)

with col1:
    Age = st.text_input('Masukkan Umur')
with col2:
    SystolicBP = st.text_input("Masukkan Nilai SystolicBP")
with col1:
    DiastolicBP = st.text_input("Masukkan Nilai DiastolicBP")
with col2:
    BS = st.text_input("Masukkan Nilai BS")
with col1:
    BodyTemp = st.text_input("Masukkan Nilai Temperatur Badan (Fahrenheit)")
with col2:
    HeartRate = st.text_input("Masukkan Nilai Heart Rate")

# Code untuk Prediksi
maternal_diagnosis = ''

# Membuat Tombol untuk Prediksi
if st.button('Test Prediksi Risiko Kesehatan pada Ibu'):
    try:
        # Konversi input ke float
        input_data = [float(Age), float(SystolicBP), float(DiastolicBP), float(BS), float(BodyTemp), float(HeartRate)]
        
        # Reshape data untuk prediksi dan standarisasi
        input_data_as_numpy_array = np.array(input_data).reshape(1, -1)
        std_data = scaler.transform(input_data_as_numpy_array)

        # Melakukan prediksi
        maternal_prediction = maternalrisk_model.predict(std_data)

        if maternal_prediction[0] == 0:
            maternal_diagnosis = 'Ibu Berisiko Tinggi'
        elif maternal_prediction[0] == 1:
            maternal_diagnosis = 'Ibu Berisiko Rendah'
        else:
            maternal_diagnosis = 'Ibu Berisiko Sedang'
        st.success(maternal_diagnosis)
    except ValueError:
        st.error("Harap masukkan semua nilai numerik dengan benar.")
