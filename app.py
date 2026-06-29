import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Konfigurasi Tampilan Web
st.set_page_config(page_title="Klasifikasi Sampah", page_icon="♻️")
st.title("♻️ Sistem Klasifikasi Sampah AI")
st.write("Aplikasi ini dibuat untuk mengklasifikasikan sampah **Organik** dan **Anorganik** menggunakan Convolutional Neural Network (CNN).")

# Fungsi untuk memuat model (di-cache agar tidak loading terus-menerus)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_sampah.h5', compile=False)

model = load_model()

# Area Upload Gambar
uploaded_file = st.file_uploader("Upload foto sampah di sini...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diupload
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    with st.spinner('AI sedang menganalisis gambar...'):
        # Preprocessing gambar (Samakan dengan proses di Colab)
        img = image.resize((150, 150)) # Ubah ukuran ke 150x150
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch
        img_array = img_array / 255.0 # Normalisasi 0-1
        
        # Proses Prediksi
        predictions = model.predict(img_array)
        
        # Penentuan Kelas (Asumsi urutan folder alfabet: 0 = Anorganik, 1 = Organik)
        class_names = ['Anorganik', 'Organik']
        
        # Mengambil nilai prediksi tertinggi
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
        
        # Tampilkan Hasil
        st.success(f"### Hasil Deteksi: {predicted_class}")
        st.info(f"Tingkat Keyakinan AI: {confidence:.2f}%")
