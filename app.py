import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Konfigurasi Tampilan Web
st.set_page_config(page_title="Klasifikasi Sampah", page_icon="♻️", layout="centered")

# Header UI
st.title("♻️ Sistem Klasifikasi Sampah AI")
st.markdown("**Kelompok 6** | Deteksi Sampah Menggunakan Deep Learning (CNN)")
st.write("---")
st.write("Aplikasi ini dibuat untuk mengklasifikasikan sampah **Organik** (mudah terurai) dan **Anorganik** (sulit terurai) secara otomatis.")

@st.cache_resource
def load_model():
    # Load model murni tanpa embel-embel legacy
    return tf.keras.models.load_model('model_sampah.h5', compile=False)

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model. Error: {e}")
    st.stop()

# Area Upload Gambar
uploaded_file = st.file_uploader("📸 Upload foto sampah di sini (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert gambar ke RGB untuk mencegah error PNG
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar yang akan dianalisis', use_column_width=True)
    
    with st.spinner('🤖 AI sedang mengekstrak fitur gambar...'):
        img = image.resize((150, 150))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0 
        
        predictions = model.predict(img_array)
        class_names = ['Organik', 'Anorganik']
        
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = np.max(predictions) * 100
        
        st.markdown("---")
        
        if predicted_class == 'Organik':
            st.success(f"### 🍃 Hasil Deteksi: **{predicted_class}**")
            st.caption("💡 **Saran:** Buang ke tempat sampah kompos. Sampah ini mudah terurai secara alami (sisa makanan, sayur, dedaunan).")
        else:
            st.info(f"### 🥫 Hasil Deteksi: **{predicted_class}**")
            st.caption("💡 **Saran:** Buang ke tempat sampah daur ulang. Sampah ini sulit terurai dan bisa dimanfaatkan kembali (plastik, kaca, logam).")
            
        st.write(f"**Tingkat Keyakinan AI:** {confidence:.2f}%")
