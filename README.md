# 🌾 SiPadi — Sistem Klasifikasi Penyakit Daun Padi

Aplikasi web berbasis *Deep Learning* untuk mendeteksi penyakit pada daun padi secara otomatis beserta visualisasi area fokus model menggunakan *Explainable AI* (**Grad-CAM**).

🔗 **Live Demo:** [https://sistem-sipadi.streamlit.app](https://sistem-sipadi.streamlit.app)

## ✨ Fitur Utama
- **Klasifikasi Otomatis (5 Kelas):** Mendeteksi *Bercak Coklat (Brown Spot)*, *Blas (Blast)*, *Hawar Daun Bakteri*, *Sehat*, dan *Lainnya*.
- **Explainable AI (Grad-CAM):** Menampilkan visualisasi *heatmap* untuk menunjukkan bagian daun yang menjadi dasar keputusan model.
- **Dukungan Multi-Format:** Mendukung unggahan gambar berformat JPG, PNG, maupun HEIC.

## 🛠️ Teknologi yang Digunakan
- **Deep Learning & Computer Vision:** PyTorch, EfficientNet, PyTorch-Grad-CAM
- **Web Framework & UI:** Streamlit
- **Image Processing:** Pillow, pillow-heif, NumPy
