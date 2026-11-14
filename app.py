import streamlit as st
from PIL import Image
from pillow_heif import register_heif_opener
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import time
import numpy as np

# ─── IMPORT LIBRARY GRAD-CAM ──────────────────────────────────────────────────
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ─── Daftarkan pembaca HEIC ───────────────────────────────────────────────────
register_heif_opener()

# ─── Konfigurasi Halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="SiPadi — Sistem Klasifikasi Penyakit Daun Padi",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Inisialisasi Session State (Algoritma Cache Gambar) ──────────────────────
if "saved_image" not in st.session_state:
    st.session_state["saved_image"] = None

def save_uploaded_file():
    if st.session_state.get("uploader") is not None:
        image = Image.open(st.session_state["uploader"])
        if image.mode != "RGB":
            image = image.convert("RGB")
        st.session_state["saved_image"] = image

def clear_image():
    st.session_state["saved_image"] = None
    if "result" in st.session_state:
        del st.session_state["result"]
    if "heatmap_img" in st.session_state:
        del st.session_state["heatmap_img"]
    if "uploader" in st.session_state:
        del st.session_state["uploader"]

# ─── CSS Kustom (DEBUGGING TINGKAT DEWA UNTUK STREAMLIT) ──────────────
st.markdown("""
<style>
/* ── Gunakan Font Sistem ── */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

/* ── Hide default Streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; }

/* ── App Background ── */
.stApp {
    background: linear-gradient(160deg, #0d3320 0%, #1a5c38 40%, #2e7d52 70%, #0d3320 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a2818 0%, #0d3320 50%, #091f13 100%) !important;
    border-right: 1px solid rgba(201,168,76,0.3);
}
[data-testid="stSidebar"] * { color: #fdf6e3 !important; }

/* ── PERBAIKAN MENU NAVIGASI KIRI ── */
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background-color: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(201, 168, 76, 0.4) !important;
    border-radius: 8px !important;
    padding: 12px 15px !important;
    margin-bottom: 10px !important;
    transition: all 0.3s ease !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label div[data-baseweb="radio"] div:first-child {
    display: none !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label p {
    color: #fdf6e3 !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background-color: rgba(201, 168, 76, 0.2) !important;
    border-color: #c9a84c !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
[data-testid="stSidebar"] div[role="radiogroup"] > label[aria-checked="true"] {
    background-color: #c9a84c !important;
    border-color: #c9a84c !important;
    box-shadow: 0 4px 10px rgba(201, 168, 76, 0.4) !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] p,
[data-testid="stSidebar"] div[role="radiogroup"] > label[aria-checked="true"] p {
    color: #0d3320 !important;
    font-weight: 900 !important;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(13,51,32,0.95) 0%, rgba(46,139,87,0.85) 100%);
    border: 1px solid rgba(201,168,76,0.4);
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner h1 {
    font-family: Georgia, serif !important;
    font-size: 2.4rem;
    font-weight: 900;
    color: #fdf6e3;
    margin: 0 0 8px 0;
}
.hero-banner .subtitle {
    color: #e8d08a;
    font-size: 1.05rem;
    margin: 0;
}

/* ── Glass Cards Umum ── */
.glass-card {
    background: rgba(255,255,255,0.95) !important;
    border: 2px solid #c9a84c !important;
    border-radius: 18px;
    padding: 28px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    margin-bottom: 20px;
}
.glass-card p, .glass-card li {
    color: #1a1a1a !important;
    line-height: 1.6;
}
.glass-card h3 {
    font-family: Georgia, serif !important;
    color: #0d3320 !important;
    margin-bottom: 12px;
    border-bottom: 2px solid #c9a84c;
    padding-bottom: 8px;
    font-weight: 900 !important;
}

/* ── Tabel ── */
table th, table th * {
    color: #ffffff !important;
    font-weight: 900 !important;
}
table td, table td * {
    color: #1a1a1a !important;
}

/* ── Upload Zone ── */
[data-testid="stFileUploadDropzone"] {
    background: rgba(255, 255, 255, 0.15) !important;
    border: 2px dashed #c9a84c !important;
    border-radius: 16px !important;
    padding: 20px !important;
}
[data-testid="stFileUploadDropzone"] div div span {
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
}
[data-testid="stFileUploadDropzone"] small {
    color: rgba(255,255,255,0.9) !important;
    font-weight: 500 !important;
}

/* ── Tombol Utama ── */
.stButton > button {
    background-color: #0d3320 !important;
    color: #ffffff !important;
    border: 2px solid #0d3320 !important;
    border-radius: 12px !important;
    padding: 14px 36px !important;
    font-weight: 900 !important;
    width: 100% !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1) !important;
}

.stButton > button:hover {
    background-color: #c9a84c !important;
    color: #000000 !important;
    border-color: #c9a84c !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(201,168,76,0.4) !important;
}

/* ── KUSTOMISASI METRIC ── */
div[data-testid="metric-container"] {
    background-color: #ffffff !important;
    border-left: 5px solid #c9a84c !important;
    border: 1px solid #e0e0e0 !important;
    padding: 15px 20px !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
div[data-testid="metric-container"] label * {
    color: #666666 !important;
    font-weight: 800 !important;
    text-transform: uppercase;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] > div {
    color: #1a5c38 !important;
    font-weight: 900 !important;
}

/* ── PERBAIKAN TABS ── */
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 12px !important;
    background-color: transparent !important;
}
.stTabs button[role="tab"] {
    background-color: #0d3320 !important;
    border: 2px solid #c9a84c !important;
    border-radius: 50px !important;
    padding: 10px 24px !important;
    margin-right: 5px !important;
    min-height: 45px !important;
    transition: all 0.3s ease !important;
}
.stTabs button[role="tab"] p {
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    margin: 0 !important;
}
.stTabs button[role="tab"]:hover {
    background-color: #1a5c38 !important;
    border-color: #e8d08a !important;
}
.stTabs button[role="tab"][aria-selected="true"] {
    background-color: #c9a84c !important;
    border-color: #c9a84c !important;
    box-shadow: 0 4px 12px rgba(201,168,76,0.4) !important;
}
.stTabs button[role="tab"][aria-selected="true"] p {
    color: #0d3320 !important;
    font-weight: 900 !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Model ───────────────────────────────────────────────────────────────
class_names = ['Bercak Coklat', 'Blas', 'Hawar', 'Lainnya', 'Sehat']
num_classes = len(class_names)

@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = models.efficientnet_v2_s(weights=None)
        in_features = 1280
    except AttributeError:
        model = models.efficientnet_b0(weights=None)
        in_features = 1280

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )

    state_dict = torch.load('best_efficientnet.pt', map_location=device)
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    return model, device

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <span style="font-size: 48px;">🌾</span>
        <h2 style="font-family: Georgia, serif; margin: 0;">SiPadi</h2>
        
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "**Navigasi Menu**",
        options=["Klasifikasi Penyakit", "Informasi Penyakit", "Ketentuan Gambar"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem; padding:0 4px;">
    <b>Kelas yang Didukung:</b><br>
    🟠 Bercak Coklat<br>
    🔴 Blas (Blast)<br>
    🟡 Hawar Daun Bakteri<br>
    🟢 Sehat<br>
    ⚪ Lainnya
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HALAMAN 1 — KLASIFIKASI PENYAKIT
# ══════════════════════════════════════════════════════════════════════════════
if "Klasifikasi" in menu:

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge" style="color:#e8d08a; font-weight:bold; font-size:14px; margin-bottom:10px;">Klasifikasi Penyakit</div>
        <h1>Sistem Klasifikasi<br>Penyakit Daun Padi</h1>
        <p class="subtitle">Unggah foto daun padi untuk mengidentifikasi penyakit</p>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="glass-card"><h3>Unggah Gambar Daun</h3></div>', unsafe_allow_html=True)

        if st.session_state["saved_image"] is None:
            st.file_uploader(
                "Format yang didukung: JPG, JPEG, PNG, HEIC",
                type=["jpg", "jpeg", "png", "heic"],
                label_visibility="visible",
                key="uploader",
                on_change=save_uploaded_file
            )
            st.info("💡 Belum ada gambar yang diunggah. Silakan pilih file gambar daun padi di atas.")
        else:
            st.image(st.session_state["saved_image"], caption="Preview Daun Padi", use_column_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.button("Hapus / Ganti Gambar", on_click=clear_image)
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Mulai Klasifikasi Sekarang"):
                with st.spinner("Model sedang menganalisis gambar..."):
                    try:
                        model, device = load_model()
                        img_tensor = transform(st.session_state["saved_image"]).unsqueeze(0).to(device)
                        
                        waktu_mulai = time.perf_counter()
                        
                        # 1. Inferensi Klasifikasi
                        with torch.no_grad():
                            output = model(img_tensor)
                            probs = F.softmax(output, dim=1)[0]
                        
                        confidence, pred_idx = torch.max(probs, 0)
                        nama_penyakit = class_names[pred_idx.item()]
                        tingkat_keyakinan = confidence.item() * 100
                        
                        waktu_selesai = time.perf_counter()
                        lama_proses = waktu_selesai - waktu_mulai

                        # 2. Proses Grad-CAM (Heatmap) 
                        # Targetkan layer konvolusi terakhir EfficientNet dengan indeks [0]
                        target_layers = [model.features[-1][0]]
                        cam = GradCAM(model=model, target_layers=target_layers)
                        targets = [ClassifierOutputTarget(pred_idx.item())]
                        
                        # Generate heatmap dengan tambahan parameter penghalus (smooth)
                        grayscale_cam = cam(input_tensor=img_tensor, 
                                            targets=targets,
                                            aug_smooth=True,
                                            eigen_smooth=True)[0, :]
                        
                        # Gabungkan gambar asli dengan heatmap
                        img_resized = st.session_state["saved_image"].resize((640, 640))
                        img_np = np.array(img_resized, dtype=np.float32) / 255.0
                        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

                        # Simpan ke Session State
                        st.session_state["result"] = {
                            "nama": nama_penyakit,
                            "conf": tingkat_keyakinan,
                            "waktu": lama_proses,
                        }
                        st.session_state["heatmap_img"] = cam_image
                        
                    except Exception as e:
                        st.error(f"❌ Terjadi kesalahan saat memproses: {e}")

    with col_result:
        st.markdown('<div class="glass-card"><h3>Hasil Klasifikasi</h3></div>', unsafe_allow_html=True)

        if "result" in st.session_state and st.session_state["saved_image"] is not None:
            r = st.session_state["result"]
            nama  = r["nama"]
            conf  = r["conf"]
            waktu = r["waktu"]

            db_info = {
                "Blas": {
                    "display": "Penyakit Blas (Blast)",
                    "info": "**Penyebab & Gejala Visual:**\nDisebabkan oleh jamur *Pyricularia grisea* yang mampu menginfeksi pada seluruh fase pertumbuhan. Gejala utama dapat dilihat dari kemunculan bercak coklat berbentuk **belah ketupat** dengan bagian tepi yang agak runcing. Bagian tengah bercak biasanya berwarna abu-abuan atau agak putih, sedangkan bagian pinggirnya berwarna coklat dengan sedikit corak oranye.\n\n**Rekomendasi Penanganan:**\nTerapkan sanitasi lahan, gunakan benih sehat, atur dosis pupuk nitrogen agar tidak berlebihan, dan terapkan sistem tanam jajar yang diberikan jarak. Jika sudah terserang, gunakan pestisida nabati atau segera aplikasikan obat kimia golongan **FUNGISIDA**."
                },
                "Bercak Coklat": {
                    "display": "Bercak Coklat (Brown Spot)",
                    "info": "**Penyebab & Gejala Visual:**\nDisebabkan oleh jamur *Bipolaris oryzae*. Tanda awal serangan adalah munculnya bercak-bercak kecil berwarna coklat gelap hingga kehitaman. Awalnya bercak tersebut berbentuk **oval atau bulat kecil**, lalu menyebar dan membesar dengan bagian tengah bercak berubah menjadi coklat terang dan tepiannya semakin gelap.\n\n**Rekomendasi Penanganan:**\nLakukan Pengendalian Hama Terpadu (PHT). Rendam benih ke dalam air panas bersuhu 50°C selama 10 menit. Berikan pupuk kalium dan silika secara seimbang untuk memperkuat struktur dinding sel. Jika serangan dalam tahap berat, lakukan penyemprotan obat golongan **FUNGISIDA** secara selektif."
                },
                "Hawar": {
                    "display": "Hawar Daun Bakteri (Kresek)",
                    "info": "**Penyebab & Gejala Visual:**\nDisebabkan oleh bakteri gram negatif *Xanthomonas oryzae*. Ditandai dengan munculnya **bercak nekrosis (kematian jaringan) pada bagian tepi daun**. Tepi daun berubah warna menjadi keabu-abuan lalu kecoklatan, namun **tulang daun seringkali masih terlihat segar**.\n\n**Rekomendasi Penanganan:**\nJangan memupuk tanaman dengan nitrogen secara berlebihan, selalu tambahkan pupuk kalium, dan terapkan sistem pengairan berselang (agar lahan tidak tergenang terus-menerus). Penyakit ini MUTLAK harus diobati dengan obat golongan **BAKTERISIDA**. Kesalahan menebak dan memberikan fungisida tidak akan mempan dan berakibat sangat fatal hingga gagal panen."
                },
                "Sehat": {
                    "display": "Daun Padi Sehat",
                    "info": "**Kondisi:**\nTanaman padi terdeteksi normal dan sehat. Tidak ada indikasi visual yang menunjukkan adanya serangan patogen jamur maupun bakteri pada jaringan daun.\n\n**Rekomendasi Perawatan:**\nTerus pertahankan praktik budidaya yang baik. Pastikan sanitasi lahan selalu dijaga serta pola pemupukan (Nitrogen, Fosfor, Kalium) diterapkan secara seimbang sesuai anjuran agar tanaman tetap kebal terhadap infeksi di kemudian hari."
                },
                "Lainnya": {
                    "display": "Tidak Dikenali (Lainnya)",
                    "info": "**Keterangan:**\nObjek dalam gambar terdeteksi **bukan daun padi** atau berada di luar cakupan penyakit yang diteliti dalam sistem klasifikasi ini.\n\n**Saran:**\nSilakan cek kembali menu **Panduan Ketentuan Gambar** untuk memastikan foto diambil dengan cara yang benar."
                }
            }

            data = db_info.get(nama, db_info["Lainnya"])

            st.success(f"✅ Analisis Selesai: **{data['display']}**")
            
            # --- TAMPILKAN HEATMAP GRAD-CAM ---
            if "heatmap_img" in st.session_state:
                st.image(st.session_state["heatmap_img"], caption="🔥 Explainable AI (Heatmap Grad-CAM)", use_column_width=True)
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric(label="Tingkat Keyakinan", value=f"{conf:.2f}%")
            # col_m2.metric(label="Waktu Inferensi", value=f"{waktu:.3f} dtk")

            st.markdown("---")
            
            if nama == "Sehat":
                st.success(data['info'])
            elif nama == "Lainnya":
                st.warning(data['info'])
            else:
                st.error(data['info']) 

        elif st.session_state["saved_image"] is not None:
            st.info("💡 Gambar siap. Klik tombol **'Mulai Klasifikasi Sekarang'** di sebelah kiri untuk melihat hasil analisis AI.")
        else:
            st.markdown("""
            <div style="text-align:center; padding:60px 20px;">
                <div style="font-size:64px; opacity:0.25;">🌾</div>
                <p style="color:rgba(30,30,30,0.4); font-size:0.9rem; margin-top:12px;">
                    Hasil klasifikasi akan muncul di sini setelah gambar dianalisis.
                </p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HALAMAN 2 — INFORMASI SINGKAT PENYAKIT
# ══════════════════════════════════════════════════════════════════════════════
elif "Informasi" in menu:

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge" style="color:#e8d08a; font-weight:bold; font-size:14px; margin-bottom:10px;">Informasi Penyakit</div>
        <h1>Informasi <br>Penyakit Daun Padi</h1>
        <p class="subtitle">Ciri-ciri, penyebab, faktor risiko, dan cara penanganan berdasarkan tinjauan pustaka</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🔴 Blas", "🟠 Bercak Coklat", "🟡 Hawar Daun", "🟢 Daun Sehat"])

    with tab1:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#c0392b !important;">🔴 Penyakit Blas (Blast)</h3>
            <p>Penyakit blas disebabkan oleh serangan jamur <em>Pyricularia grisea</em> yang mampu menginfeksi tanaman padi pada seluruh fase pertumbuhan, mulai dari masa persemaian benih hingga menjelang waktu panen.</p>
            <ul>
                <li><b>Gejala Visual Utama:</b> Munculnya bercak coklat berbentuk <b>belah ketupat</b> dengan bagian tepi yang agak runcing. Pada bagian tengah bercak tersebut biasanya berwarna abu-abuan atau agak putih, sedangkan bagian pinggirnya berwarna coklat dengan sedikit corak oranye.</li>
                <li><b>Faktor Pemicu:</b> Kondisi lingkungan yang mendukung, seperti tingkat kelembapan yang tinggi dan suhu udara yang hangat berkisar antara <b>25–28°C</b>. Selain itu dipicu oleh curah hujan tinggi, jarak tanam yang terlalu rapat, penggunaan varietas rentan, serta pemupukan unsur nitrogen yang berlebihan.</li>
                <li><b>Penanganan & Pencegahan:</b> Pencegahan dilakukan melalui sanitasi lahan, benih sehat, dan sistem tanam jajar. Jika sudah terlanjur terserang, langkah pengendaliannya adalah menggunakan pestisida nabati (bawang merah, bawang putih, kunyit, lengkuas) atau pengobatan <b>FUNGISIDA</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#c06010 !important;">🟠 Bercak Coklat (Brown Spot)</h3>
            <p>Penyakit bercak coklat disebabkan oleh jamur <em>Bipolaris oryzae</em> (atau <em>Helminthosporium oryzae</em>). Penyakit ini sangat merugikan, terutama apabila menyerang lahan sawah yang kekurangan unsur hara.</p>
            <ul>
                <li><b>Sejarah:</b> Memiliki sejarah kelam dan menjadi penyebab utama bencana kelaparan di Benggala, India, pada tahun 1942.</li>
                <li><b>Gejala Visual Utama:</b> Tanda awal serangan adalah munculnya bercak-bercak kecil berwarna coklat gelap hingga kehitaman pada permukaan daun. Awalnya bercak berbentuk <b>oval atau bulat kecil</b>, lalu perlahan menyebar dengan bagian tengah berubah menjadi coklat terang dan tepian semakin gelap.</li>
                <li><b>Penanganan & Pencegahan:</b> Dilakukan melalui Pengendalian Hama Terpadu (PHT). Secara mekanis, rendam benih di dalam air panas <b>50°C selama 10 menit</b> untuk mematikan spora. Diperlukan juga pengelolaan nutrisi tanah seperti pemberian pupuk <b>kalium dan silika</b> secara seimbang. Jika parah, gunakan <b>FUNGISIDA</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color:#b5770d !important;">🟡 Hawar Daun Bakteri (HDB / Kresek)</h3>
            <p>Disebabkan oleh bakteri gram negatif <em>Xanthomonas oryzae</em> yang mampu menyerang tanaman pada seluruh fase pertumbuhan.</p>
            <ul>
                <li><b>Gejala Visual Utama:</b> Ditandai dengan munculnya <b>bercak nekrosis (kematian jaringan) pada bagian tepi daun</b>. Kerusakan selalu berawal dari tepi daun yang berubah warna menjadi keabu-abuan/kecoklatan. Ciri khas yang mengecoh: <b>tulang daun seringkali masih terlihat segar</b>. Pada tanaman muda disebut <em>kresek</em> (layu mati), pada tanaman dewasa disebut <em>hawar (blight)</em>.</li>
                <li><b>Faktor Pemicu:</b> Tingkat kelembapan udara tinggi pada saat <b>musim hujan</b>. Pemberian pupuk nitrogen dengan dosis tinggi tanpa diimbangi oleh pupuk kalium.</li>
                <li><b>Penanganan & Pengobatan:</b> Penyakit ini disebabkan oleh bakteri, oleh karena itu <b>MUTLAK harus diobati dengan BAKTERISIDA</b>. Penggunaan fungisida merupakan kesalahan diagnosis yang sangat fatal dan tidak akan mempan. Terapkan sistem pengairan berselang agar lahan sawah tidak tergenang air secara terus-menerus.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.markdown("""
        <div class="glass-card" style="border-left: 8px solid #1a7a45;">
            <h3 style="color:#1a7a45 !important;">🟢 Daun Padi Sehat</h3>
            <p>Kondisi ideal tanaman padi yang terbebas dari infeksi patogen yang merusak jaringan (baik jamur maupun bakteri).</p>
            <ul>
                <li><b>Ciri-Ciri:</b> Warna daun hijau segar secara merata, tidak ada kemunculan bercak, perubahan warna, atau nekrosis pada bagian tengah maupun tepi daun.</li>
                <li><b>Saran Perawatan:</b> Terus pertahankan sanitasi lahan, gunakan benih padi dari varietas unggul yang tahan penyakit, serta menjaga pola pemupukan yang seimbang agar daun padi memiliki kekebalan alami.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header" style="margin-top:10px">
        <h2>Tabel Perbandingan Karakteristik Penyakit</h2>
        <div class="section-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
    <table style="width:100%; border-collapse:collapse; font-size:0.88rem; color:#1a1a1a !important;">
        <thead>
            <tr style="background:linear-gradient(135deg,#0d3320,#1a5c38);">
                <th style="padding:12px 14px; text-align:left; border-radius:8px 0 0 0; color:#ffffff !important;">Aspek</th>
                <th style="padding:12px 14px; text-align:center; color:#ffffff !important;">🔴 Blas (Blast)</th>
                <th style="padding:12px 14px; text-align:center; color:#ffffff !important;">🟠 Bercak Coklat</th>
                <th style="padding:12px 14px; text-align:center; border-radius:0 8px 0 0; color:#ffffff !important;">🟡 Hawar (HDB)</th>
            </tr>
        </thead>
        <tbody>
            <tr style="background:#fafafa;">
                <td style="padding:11px 14px; font-weight:600; color:#2e8b57 !important;">Penyebab (Patogen)</td>
                <td style="padding:11px 14px; text-align:center;">Jamur <em>Pyricularia grisea</em></td>
                <td style="padding:11px 14px; text-align:center;">Jamur <em>Bipolaris oryzae</em></td>
                <td style="padding:11px 14px; text-align:center;">Bakteri Gram Negatif <em>Xanthomonas oryzae</em></td>
            </tr>
            <tr style="background:#fff;">
                <td style="padding:11px 14px; font-weight:600; color:#2e8b57 !important;">Karakteristik Visual</td>
                <td style="padding:11px 14px; text-align:center;">Bercak belah ketupat, ujung runcing</td>
                <td style="padding:11px 14px; text-align:center;">Bercak oval atau bulat kecil</td>
                <td style="padding:11px 14px; text-align:center;">Nekrosis di tepi daun, tulang daun segar</td>
            </tr>
            <tr style="background:#fafafa;">
                <td style="padding:11px 14px; font-weight:600; color:#2e8b57 !important;">Faktor Lingkungan Pemicu</td>
                <td style="padding:11px 14px; text-align:center;">Suhu 25–28°C, jarak tanam rapat</td>
                <td style="padding:11px 14px; text-align:center;">Lahan kurang unsur hara</td>
                <td style="padding:11px 14px; text-align:center;">Musim hujan tinggi, lahan tergenang terus</td>
            </tr>
            <tr style="background:#fff;">
                <td style="padding:11px 14px; font-weight:600; color:#2e8b57 !important;">Jenis Pengobatan Utama</td>
                <td style="padding:11px 14px; text-align:center; font-weight:700; color:#c0392b !important;">FUNGISIDA</td>
                <td style="padding:11px 14px; text-align:center; font-weight:700; color:#c0392b !important;">FUNGISIDA</td>
                <td style="padding:11px 14px; text-align:center; font-weight:700; color:#b5770d !important;">BAKTERISIDA</td>
            </tr>
            <tr style="background:#fafafa;">
                <td style="padding:11px 14px; font-weight:600; color:#2e8b57 !important;">Tindakan Tambahan</td>
                <td style="padding:11px 14px; text-align:center;">Pestisida nabati (bawang/kunyit)</td>
                <td style="padding:11px 14px; text-align:center;">Rendam benih 50°C, pupuk kalium/silika</td>
                <td style="padding:11px 14px; text-align:center;">Pupuk kalium, pengairan berselang</td>
            </tr>
        </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HALAMAN 3 — PANDUAN KETENTUAN GAMBAR
# ══════════════════════════════════════════════════════════════════════════════
elif "Ketentuan" in menu:

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge" style="color:#e8d08a; font-weight:bold; font-size:14px; margin-bottom:10px;">Ketentuan Gambar</div>
        <h1>Panduan<br>Ketentuan Gambar</h1>
        <p class="subtitle">Pastikan gambar yang diunggah sesuai standar agar hasil identifikasi AI</p>
    </div>
    """, unsafe_allow_html=True)

    col_panduan1, col_panduan2 = st.columns([1, 1], gap="large")

    with col_panduan1:
        st.markdown("""
        <div class="glass-card" style="border-top: 5px solid #2e8b57 !important;">
            <h3 style="color:#2e8b57 !important;">✅ Cara yang Benar</h3>
            <p>Agar model klasifikasi dapat bekerja maksimal, perhatikan hal-hal berikut:</p>
            <ul>
                <li><b>Fokus pada Daun:</b> Pastikan daun yang sakit berada tepat di tengah (center) dan menutupi sebagian besar frame kamera.</li>
                <li><b>Pencahayaan Cukup:</b> Ambil foto di siang hari atau pastikan daun tidak tertutup bayangan.</li>
                <li><b>Latar Belakang Netral:</b> Usahakan latar belakang tidak terlalu ramai (gunakan telapak tangan atau kertas sebagai alas).</li>
                <li><b>Gambar Tajam:</b> Pastikan kamera fokus (tidak blur) sehingga tekstur bercak penyakit terlihat jelas.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            img_benar = Image.open("benar.jpg")
            st.image(img_benar, caption="✅ Contoh Pengambilan Gambar yang Benar", use_column_width=True)
        except FileNotFoundError:
            st.info("⚠️ File 'benar.jpg' belum tersedia di direktori web Anda.")

    with col_panduan2:
        st.markdown("""
        <div class="glass-card" style="border-top: 5px solid #c0392b !important;">
            <h3 style="color:#c0392b !important;">❌ Cara yang Salah</h3>
            <p>Hindari kesalahan umum berikut saat memotret daun:</p>
            <ul>
                <li><b>Terlalu Jauh:</b> Objek daun terlalu kecil sehingga pola penyakit tidak terlihat.</li>
                <li><b>Blur/Goyang:</b> Lensa tidak fokus membuat fitur penyakit hilang dan tidak bisa dideteksi.</li>
                <li><b>Latar Belakang Rumit:</b> Memotret segerombolan padi dari jauh membuat AI kebingungan menentukan daun mana yang harus difokuskan.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # --- Kode yang ditambahkan untuk gambar salah.jpg ---
        try:
            img_salah = Image.open("salah.jpg")
            st.image(img_salah, caption="❌ Contoh Pengambilan Gambar yang Salah", use_column_width=True)
        except FileNotFoundError:
            st.info("⚠️ File 'salah.jpg' belum tersedia di direktori web Anda.")