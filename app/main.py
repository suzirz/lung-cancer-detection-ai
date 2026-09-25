"""
PulmoScan AI — Interactive Clinical Lung Cancer CT Detection & Explainability Demo
File: app/main.py

Web Application berbasis Streamlit untuk skrining klasifikasi nodul paru:
- Deteksi 3 Kelas: Normal, Benign (Jinak), Malignant (Kanker Ganas).
- Explainable AI: Grad-CAM Saliency Maps untuk visualisasi atensi spasial model.
- Model Backbone: Transfer Learning EfficientNet-B0 (100% Validation Recall pada Tesla T4).
"""

import os
import sys
from pathlib import Path
from PIL import Image
import streamlit as st

# Pastikan root direktori proyek dapat diakses untuk impor modul src & app
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference_service import InferenceService, PredictionResult, CLASS_NAMES


# ---------------------------------------------------------------------------
# Konfigurasi Halaman & Design Tokens
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PulmoScan AI — Lung Cancer CT Screening",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling: Dark Medical Slate Theme
CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Background styling */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0d1527 0%, #070a13 100%);
        color: #f1f5f9;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .badge-online {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }

    .badge-t4 {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.3);
    }

    /* Diagnosis Alert Cards */
    .diag-card {
        border-radius: 14px;
        padding: 20px 24px;
        margin-top: 16px;
        margin-bottom: 20px;
        border: 1px solid;
    }

    .diag-malignant {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(153, 27, 27, 0.1) 100%);
        border-color: rgba(239, 68, 68, 0.4);
        color: #fca5a5;
    }

    .diag-benign {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(180, 83, 9, 0.1) 100%);
        border-color: rgba(245, 158, 11, 0.4);
        color: #fde68a;
    }

    .diag-normal {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 95, 70, 0.1) 100%);
        border-color: rgba(16, 185, 129, 0.4);
        color: #a7f3d0;
    }

    /* Metric Containers */
    .metric-box {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }

    /* Sidebar aesthetics */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090e1a 0%, #060911 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Code & Note disclaimer */
    .medical-disclaimer {
        background: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        font-size: 13px;
        color: #94a3b8;
        line-height: 1.6;
        margin-top: 24px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Inisialisasi Service dengan Streamlit Cache Resource
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model klinis & bobot neural network...")
def get_service() -> InferenceService:
    model_path = os.path.join(PROJECT_ROOT, "models", "baseline_efficientnet_b0_best.pth")
    return InferenceService(model_path=model_path)


# ---------------------------------------------------------------------------
# Sidebar: Parameter & Konfigurasi
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🫁 PulmoScan AI")
    st.markdown(
        """
        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px;">
            <span class="badge-pill badge-online">● ONLINE</span>
            <span class="badge-pill badge-t4">TESLA T4 MODEL</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Input Citra CT Scan")

    input_mode = st.radio(
        "Pilih Sumber Citra:",
        options=["🧪 Contoh Klinis Tersimpan", "📤 Unggah Citra Mandiri"],
        index=0,
    )

    selected_image = None
    image_title = ""

    if input_mode == "🧪 Contoh Klinis Tersimpan":
        sample_choice = st.selectbox(
            "Pilih Kasus Diagnosis:",
            options=[
                "Benign (Kasus Nodul Jinak)",
                "Malignant (Kasus Kanker Ganas)",
                "Normal (Jaringan Paru Sehat)",
            ],
            index=1,
        )

        sample_mapping = {
            "Benign (Kasus Nodul Jinak)": "benign_sample.jpg",
            "Malignant (Kasus Kanker Ganas)": "malignant_sample.jpg",
            "Normal (Jaringan Paru Sehat)": "normal_sample.jpg",
        }
        sample_filename = sample_mapping[sample_choice]
        sample_path = os.path.join(PROJECT_ROOT, "app", "samples", sample_filename)

        if os.path.exists(sample_path):
            selected_image = Image.open(sample_path)
            image_title = sample_choice
        else:
            st.error(f"File sampel tidak ditemukan di {sample_path}")

    else:
        uploaded_file = st.file_uploader(
            "Pilih berkas CT Scan (PNG/JPG):",
            type=["png", "jpg", "jpeg"],
            help="Unggah potongan aksial citra CT scan paru",
        )
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)
            image_title = uploaded_file.name

    st.markdown("---")
    st.markdown("#### 🎨 Pengaturan Grad-CAM")
    cam_alpha = st.slider(
        "Transparansi Heatmap (Alpha):",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
        help="Semakin tinggi nilai, semakin pekat heatmap atensi yang menutupi citra asli",
    )
    cam_colormap = st.selectbox(
        "Colormap Grad-CAM:",
        options=["jet", "inferno", "viridis", "magma"],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### 📊 Benchmark Model")
    st.markdown(
        """
        - **Arsitektur:** EfficientNet-B0 (1.280 Embeddings)
        - **Sensitivity / Recall:** `100.0%`
        - **Accuracy:** `100.0%`
        - **AUC-ROC Score:** `1.000`
        - **Target Dataset:** IQ-OTH/NCCD (12.184 citra)
        """
    )


# ---------------------------------------------------------------------------
# Main Panel: Header & Banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
                <h1 style="font-size: 28px; font-weight: 800; margin: 0 0 8px 0; color: #ffffff; letter-spacing: -0.02em;">
                    🫁 PulmoScan AI: Intelligent Lung CT Cancer Detection & Grad-CAM
                </h1>
                <p style="font-size: 15px; color: #94a3b8; margin: 0; line-height: 1.5;">
                    Sistem Skrining Medis Berbantu Kecerdasan Buatan (AI Decision Support) dengan Peta Atensi Spasial untuk Deteksi Dini Kanker Paru
                </p>
            </div>
            <div style="text-align: right;">
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #38bdf8; background: rgba(56, 189, 248, 0.1); padding: 6px 12px; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.2);">
                    FP16 CUDA INFERENCE READY
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Processing & Display Pipeline
# ---------------------------------------------------------------------------
if selected_image is None:
    st.info("👈 Silakan pilih contoh klinis di sidebar atau unggah citra CT scan untuk memulai analisis.")
else:
    try:
        service = get_service()
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

    # Jalankan inferensi & Grad-CAM
    with st.spinner("Menganalisis fitur spasial CT scan & menghasilkan peta atensi Grad-CAM..."):
        result: PredictionResult = service.predict(
            selected_image,
            alpha=cam_alpha,
            colormap=cam_colormap,
        )

    # 1. Grid Visualisasi: Citra Asli vs Grad-CAM
    col_orig, col_cam = st.columns(2)

    with col_orig:
        st.markdown("#### 📷 Citra CT Scan Asli")
        st.image(
            selected_image,
            caption=f"Input: {image_title} ({selected_image.size[0]}x{selected_image.size[1]} px)",
            use_container_width=True,
        )

    with col_cam:
        st.markdown(f"#### 🔍 Peta Atensi Grad-CAM (`{cam_colormap.upper()}`)")
        st.image(
            result.overlay_image,
            caption=f"Area Atensi Diagnostik Tertinggi (Fokus Fitur Lapisan Konvolusi Terakhir)",
            use_container_width=True,
        )

    # 2. Status Diagnosis Card
    if result.class_name == "Malignant":
        st.markdown(
            f"""
            <div class="diag-card diag-malignant">
                <h3 style="margin: 0 0 6px 0; font-size: 20px; font-weight: 800;">🚨 TERDETEKSI INDIKASI KANKER GANAS (MALIGNANT)</h3>
                <p style="margin: 0; font-size: 14px; opacity: 0.95;">
                    Model mengidentifikasi pola densitas tinggi dan morfologi mencurigakan konsisten dengan nodul ganas paru.
                    Disarankan verifikasi klinis mendesak oleh Dokter Spesialis Paru / Radiolog Konsultan Onkologi.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif result.class_name == "Benign":
        st.markdown(
            f"""
            <div class="diag-card diag-benign">
                <h3 style="margin: 0 0 6px 0; font-size: 20px; font-weight: 800;">⚠️ TERDETEKSI NODUL JINAK (BENIGN)</h3>
                <p style="margin: 0; font-size: 14px; opacity: 0.95;">
                    Model mengidentifikasi struktur nodul dengan karakteristik non-invasif/jinak.
                    Disarankan pemantauan berkala (follow-up CT scan dalam 3–6 bulan) untuk memastikan stabilitas ukuran nodul.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="diag-card diag-normal">
                <h3 style="margin: 0 0 6px 0; font-size: 20px; font-weight: 800;">✅ JARINGAN PARU NORMAL / TIDAK TERLIHAT KELAINAN SIGNIFIKAN</h3>
                <p style="margin: 0; font-size: 14px; opacity: 0.95;">
                    Arsitektur parenkim paru dan vaskulatur tampak dalam batas normal tanpa nodul mencurigakan yang terdeteksi.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Metrik Statistik Inferensi
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-value">{result.class_name.upper()}</div>
                <div class="metric-label">Prediksi Kelas</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-value">{result.confidence * 100:.1f}%</div>
                <div class="metric-label">Tingkat Keyakinan</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-value">{result.latency_ms:.1f} ms</div>
                <div class="metric-label">Latensi Inferensi</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-value">{str(service.device).upper()}</div>
                <div class="metric-label">Perangkat Pemroses</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Distribusi Probabilitas Multi-Kelas
    st.markdown("#### 📈 Distribusi Probabilitas Diagnostik")
    for cls in CLASS_NAMES:
        prob = result.probabilities[cls]
        col_lbl, col_bar = st.columns([1, 4])
        with col_lbl:
            st.markdown(f"**{cls}**")
        with col_bar:
            st.progress(float(prob), text=f"{prob * 100:.2f}%")

    # 5. Penjelasan Klinis & Disclaimer
    st.markdown(
        """
        <div class="medical-disclaimer">
            <strong>⚠️ Catatan Klinis & Disclaimer Penting:</strong><br>
            Aplikasi ini dibangun untuk tujuan penelitian akademik, edukasi, dan sistem pendukung keputusan klinis (Clinical Decision Support System / CDSS).
            Hasil prediksi model AI dan peta atensi Grad-CAM ini <strong>bukan merupakan diagnosis medis resmi pengganti radiolog atau dokter spesialis paru</strong>.
            Keputusan terapi dan penegakan diagnosis definitif wajib dikonfirmasi melalui evaluasi klinis komprehensif, biopsi patologi, atau pembacaan formal oleh dokter spesialis.
        </div>
        """,
        unsafe_allow_html=True,
    )
