import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Konfigurasi Tampilan Web
st.set_page_config(page_title="Klasifikasi Sampah", page_icon="♻️")
st.title("♻️ Sistem Klasifikasi Sampah AI")
st.write("Aplikasi ini dibuat untuk mengklasifikasikan sampah **Organik** dan **Anorganik** menggunakan Convolutional Neural Network (CNN).")

# Trik Anti-Error: Bangun kerangka modelnya di sini, lalu masukkan bobotnya!
@st.cache_resource
def load_model():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(150, 150, 3)),
        tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax")
    ])
    
    # Hanya me-load bobotnya saja (sangat aman dari error)
    model.load_weights('model_sampah.h5')
    return model

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
