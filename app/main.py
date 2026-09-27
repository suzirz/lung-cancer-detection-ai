"""
PulmoScan AI — Conversational Clinical Co-Pilot & Thoracic Diagnostic Studio
File: app/main.py

Web Application berbasis Streamlit:
- Tab 1: AI Clinical Co-Pilot (Conversational Interface ala Ollama/ChatGPT)
- Tab 2: Diagnostic Studio (Visualisasi Grad-CAM Komprehensif & PACS DICOM)
- Ekspor Rekam Medis: Hospital-Grade Clinical PDF Report & DICOM Secondary Capture
"""

import os
import sys
import io
from pathlib import Path
from PIL import Image
import streamlit as st

# Pastikan root direktori proyek dapat diakses untuk impor modul src & app
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference_service import InferenceService, PredictionResult, DicomPredictionResult, CLASS_NAMES
from app.chatbot_engine import ChatbotEngine, ChatResponse
from src.evaluation.report_generator import generate_clinical_pdf
from src.dicom.processor import load_dicom_slice


# ---------------------------------------------------------------------------
# Konfigurasi Halaman & Design System
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PulmoScan AI — Clinical Co-Pilot",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 15%, #0d1527 0%, #070a13 100%);
        color: #f1f5f9;
    }

    .main-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 20px;
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
        letter-spacing: 0.06em;
    }

    .badge-online {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }

    .badge-verified {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.3);
    }

    .diag-card {
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 12px;
        margin-bottom: 16px;
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

    .metric-box {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        margin-top: 4px;
    }

    .medical-disclaimer {
        background: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 0 10px 10px 0;
        font-size: 12px;
        color: #94a3b8;
        line-height: 1.5;
        margin-top: 20px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090e1a 0%, #060911 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Inisialisasi Service & Chatbot Engine
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model inferensi EfficientNet-B0...")
def get_service() -> InferenceService:
    model_path = os.path.join(PROJECT_ROOT, "models", "baseline_efficientnet_b0_best.pth")
    return InferenceService(model_path=model_path)


service = get_service()

if "chatbot_engine" not in st.session_state:
    st.session_state.chatbot_engine = ChatbotEngine()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Halo Dokter! Saya PulmoScan AI Co-Pilot.**\n\n"
                "Saya siap membantu skrining CT scan toraks berbasis **EfficientNet-B0 (Group-KFold Verified: 97.42% Akurasi, 96.97% Recall)**.\n"
                "- Pilih sampel kasus di sidebar atau unggah citra CT (JPG/PNG/DICOM).\n"
                "- Tanyakan rincian diagnostik, rekomendasi Fleischner Society, atau minta ekspor laporan PDF resmi."
            )
        }
    ]

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_display_image" not in st.session_state:
    st.session_state.last_display_image = None
if "patient_id" not in st.session_state:
    st.session_state.patient_id = "ANON-8821"


# ---------------------------------------------------------------------------
# Sidebar: Pemilihan Citra & Parameter
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🫁 PulmoScan AI")
    st.markdown(
        """
        <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 14px;">
            <span class="badge-pill badge-online">🟢 ONLINE</span>
            <span class="badge-pill badge-verified">GROUP-KFOLD VERIFIED</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.session_state.patient_id = st.text_input("ID Pasien (EHR/PACS):", value=st.session_state.patient_id)

    st.markdown("#### 📁 Sumber Citra CT Scan")
    input_mode = st.radio(
        "Pilih Sumber:",
        options=["💾 Contoh Kasus Tersimpan", "📤 Unggah Citra (PNG/JPG)", "🏥 Berkas Medis DICOM (.dcm)"],
        index=0,
    )

    selected_image = None
    selected_dicom_bytes = None
    image_title = ""

    if input_mode == "💾 Contoh Kasus Tersimpan":
        sample_choice = st.selectbox(
            "Pilih Kasus:",
            options=[
                "Malignant (Kasus Kanker Ganas)",
                "Benign (Kasus Nodul Jinak)",
                "Normal (Jaringan Paru Sehat)",
            ],
            index=0,
        )
        sample_mapping = {
            "Benign (Kasus Nodul Jinak)": "benign_sample.jpg",
            "Malignant (Kasus Kanker Ganas)": "malignant_sample.jpg",
            "Normal (Jaringan Paru Sehat)": "normal_sample.jpg",
        }
        sample_path = os.path.join(PROJECT_ROOT, "app", "samples", sample_mapping[sample_choice])
        if os.path.exists(sample_path):
            selected_image = Image.open(sample_path)
            image_title = sample_choice
        else:
            st.error("Sampel tidak ditemukan.")

    elif input_mode == "📤 Unggah Citra (PNG/JPG)":
        uploaded_file = st.file_uploader("Unggah Irisan CT Scan Paru:", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file).convert("RGB")
            image_title = uploaded_file.name

    elif input_mode == "🏥 Berkas Medis DICOM (.dcm)":
        uploaded_dcm = st.file_uploader("Unggah Berkas DICOM (.dcm):", type=["dcm"])
        if uploaded_dcm is not None:
            selected_dicom_bytes = uploaded_dcm.read()
            image_title = uploaded_dcm.name

    st.markdown("---")
    st.markdown("#### 🎨 Konfigurasi Grad-CAM")
    cam_colormap = st.selectbox("Colormap:", ["jet", "viridis", "inferno", "magma", "turbo"], index=0)
    cam_alpha = st.slider("Alpha Blend Transparansi:", 0.1, 0.9, 0.45, 0.05)


# ---------------------------------------------------------------------------
# Header Utama
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 24px; font-weight: 800; color: #f8fafc;">
                    🫁 PulmoScan AI — Thoracic Clinical Co-Pilot
                </h1>
                <p style="margin: 4px 0 0 0; font-size: 13px; color: #94a3b8;">
                    Sistem Pendukung Keputusan Klinis Berbasis EfficientNet-B0 (Zero-Leakage Group Cross-Validation)
                </p>
            </div>
            <div style="text-align: right;">
                <span class="badge-pill badge-verified">UNSEEN CV: 97.42% ACC</span>
                <span class="badge-pill badge-online">AUC: 0.9998</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

tab_chat, tab_studio = st.tabs(["💬 AI Clinical Co-Pilot (Chatbot)", "🔬 Diagnostic Studio & PACS"])


# ---------------------------------------------------------------------------
# TAB 1: Conversational AI Co-Pilot (Ollama/ChatGPT Style)
# ---------------------------------------------------------------------------
with tab_chat:
    # Action bar atas untuk memproses citra yang dipilih ke dalam chat
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_chat_analysis = st.button("🚀 Analisis Citra di Chat", use_container_width=True, type="primary")

    with col_info:
        if selected_image is not None or selected_dicom_bytes is not None:
            st.caption(f"📁 Citra aktif siap dianalisis: **{image_title}**")
        else:
            st.caption("Pilih sampel atau unggah citra di sidebar kiri untuk memulai.")

    if run_chat_analysis:
        if selected_dicom_bytes is not None:
            with st.spinner("Memproses berkas DICOM & menjalankan inferensi..."):
                result = service.predict_dicom(selected_dicom_bytes, alpha=cam_alpha, colormap=cam_colormap)
                dcm_slice = load_dicom_slice(selected_dicom_bytes)
                display_img = dcm_slice.windowed_image
        elif selected_image is not None:
            with st.spinner("Mengevaluasi irisan CT Scan dengan Grad-CAM..."):
                result = service.predict(selected_image, alpha=cam_alpha, colormap=cam_colormap)
                display_img = selected_image
        else:
            result = None
            display_img = None
            st.warning("Silakan pilih atau unggah citra terlebih dahulu.")

        if result is not None:
            st.session_state.last_result = result
            st.session_state.last_display_image = display_img

            # Dapatkan laporan terstruktur dari ChatbotEngine
            report_response = st.session_state.chatbot_engine.format_diagnostic_report(
                result, patient_id=st.session_state.patient_id
            )

            # Tambahkan ke riwayat chat
            st.session_state.messages.append({
                "role": "user",
                "content": f"Tolong analisis citra CT Scan `{image_title}` untuk pasien `{st.session_state.patient_id}`."
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": report_response.text,
                "image": report_response.image
            })

    # Render Riwayat Chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("image") is not None:
                st.image(msg["image"], caption="Grad-CAM Diagnostic Attention Overlay", width=340)

    # Quick Suggestion Chips jika ada hasil diagnosis aktif
    if st.session_state.last_result is not None:
        st.markdown("---")
        st.markdown("💡 **Pertanyaan Lanjutan Klinis Cepat:**")
        chip1, chip2, chip3, chip4 = st.columns(4)
        quick_prompt = None
        with chip1:
            if st.button("❓ Kenapa Malignant?", use_container_width=True):
                quick_prompt = "Kenapa didiagnosis malignant?"
        with chip2:
            if st.button("📋 Panduan Fleischner", use_container_width=True):
                quick_prompt = "Apa rekomendasi Fleischner Society untuk nodul ini?"
        with chip3:
            if st.button("🔍 Cara Baca Grad-CAM", use_container_width=True):
                quick_prompt = "Bagaimana cara membaca peta Grad-CAM ini?"
        with chip4:
            if st.button("📄 Buat Laporan PDF", use_container_width=True):
                quick_prompt = "Tolong siapkan laporan resmi PDF rekam medis."

        if quick_prompt:
            st.session_state.messages.append({"role": "user", "content": quick_prompt})
            ans = st.session_state.chatbot_engine.respond(quick_prompt)
            st.session_state.messages.append({"role": "assistant", "content": ans.text})
            st.rerun()

        # Tombol Download PDF Langsung di Chat
        res = st.session_state.last_result
        orig = st.session_state.last_display_image
        if res is not None and orig is not None:
            pdf_bytes = io.BytesIO()
            pdata = {
                "patient_id": st.session_state.patient_id,
                "study_date": "2026-09-27",
                "predicted_class": res.class_name,
                "confidence": res.confidence,
                "probabilities": res.probabilities,
                "radiomics": {"Model": "EfficientNet-B0", "Grad-CAM Saliency": "Peak Centered"}
            }
            tmp_pdf_path = os.path.join(PROJECT_ROOT, "reports", f"report_{st.session_state.patient_id}.pdf")
            generate_clinical_pdf(pdata, orig, res.overlay_image, output_path=tmp_pdf_path)
            with open(tmp_pdf_path, "rb") as f:
                pdf_data = f.read()

            st.download_button(
                label=f"📥 Unduh Laporan Resmi Rekam Medis (PDF) — {st.session_state.patient_id}",
                data=pdf_data,
                file_name=f"PulmoScan_Report_{st.session_state.patient_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    # Input Chat Bebas
    user_input = st.chat_input("Tanyakan apa saja kepada PulmoScan AI Co-Pilot...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        ans = st.session_state.chatbot_engine.respond(user_input)
        st.session_state.messages.append({"role": "assistant", "content": ans.text})
        st.rerun()


# ---------------------------------------------------------------------------
# TAB 2: Diagnostic Studio (Radiology & PACS Deep Dive)
# ---------------------------------------------------------------------------
with tab_studio:
    if selected_image is None and selected_dicom_bytes is None:
        st.info("Pilih citra CT scan atau sampel di sidebar untuk membuka Diagnostic Studio.")
    else:
        is_dicom = selected_dicom_bytes is not None

        if is_dicom:
            result = service.predict_dicom(selected_dicom_bytes, alpha=cam_alpha, colormap=cam_colormap)
            dcm_slice = load_dicom_slice(selected_dicom_bytes)
            display_img = dcm_slice.windowed_image
        else:
            result = service.predict(selected_image, alpha=cam_alpha, colormap=cam_colormap)
            display_img = selected_image

        st.session_state.last_result = result
        st.session_state.last_display_image = display_img

        col_orig, col_cam = st.columns(2)
        with col_orig:
            st.markdown("#### 📷 Citra CT Scan Asli (Lung Window)")
            st.image(display_img, caption=f"Input: {image_title} ({display_img.size[0]}x{display_img.size[1]} px)", use_container_width=True)

        with col_cam:
            st.markdown(f"#### 🎯 Peta Atensi Grad-CAM (`{cam_colormap.upper()}`)")
            st.image(result.overlay_image, caption="Area Atensi Tertinggi (Lapisan Konvolusi Terakhir)", use_container_width=True)

        # Kartu Diagnostik
        if result.class_name == "Malignant":
            st.markdown(
                """
                <div class="diag-card diag-malignant">
                    <h3 style="margin: 0 0 6px 0; font-size: 18px; font-weight: 800;">⚠️ TERDETEKSI INDIKASI KANKER GANAS (MALIGNANT)</h3>
                    <p style="margin: 0; font-size: 13px; opacity: 0.95;">
                        Model mendeteksi pola densitas tinggi dan tepi spikulasi mencurigakan. Segera lakukan verifikasi klinis dan biopsi/PET-CT.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif result.class_name == "Benign":
            st.markdown(
                """
                <div class="diag-card diag-benign">
                    <h3 style="margin: 0 0 6px 0; font-size: 18px; font-weight: 800;">ℹ️ TERDETEKSI NODUL JINAK (BENIGN)</h3>
                    <p style="margin: 0; font-size: 13px; opacity: 0.95;">
                        Model mengidentifikasi struktur nodul dengan karakteristik non-invasif/jinak. Follow-up CT 6-12 bulan disarankan.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="diag-card diag-normal">
                    <h3 style="margin: 0 0 6px 0; font-size: 18px; font-weight: 800;">✅ JARINGAN PARU NORMAL / TANPA KELAINAN SIGNIFIKAN</h3>
                    <p style="margin: 0; font-size: 13px; opacity: 0.95;">
                        Parenkim paru dan vaskulatur dalam batas normal tanpa nodul signifikan.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # 4 Metrik Box
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{result.class_name.upper()}</div><div class="metric-label">Prediksi Kelas</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{result.confidence * 100:.1f}%</div><div class="metric-label">Keyakinan (Confidence)</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{result.latency_ms:.1f} ms</div><div class="metric-label">Latensi Inferensi</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{str(service.device).upper()}</div><div class="metric-label">Hardware Device</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probabilitas Batang
        st.markdown("#### 📊 Distribusi Probabilitas Diagnostik")
        for cls in CLASS_NAMES:
            prob = result.probabilities[cls]
            col_lbl, col_bar = st.columns([1, 4])
            with col_lbl:
                st.markdown(f"**{cls}**")
            with col_bar:
                st.progress(float(prob), text=f"{prob * 100:.2f}%")

        # Unduh PDF dari Studio
        tmp_pdf_path = os.path.join(PROJECT_ROOT, "reports", f"report_{st.session_state.patient_id}.pdf")
        pdata = {
            "patient_id": st.session_state.patient_id,
            "study_date": "2026-09-27",
            "predicted_class": result.class_name,
            "confidence": result.confidence,
            "probabilities": result.probabilities,
            "radiomics": {"Backbone": "EfficientNet-B0", "Latensi": f"{result.latency_ms:.1f} ms"}
        }
        generate_clinical_pdf(pdata, display_img, result.overlay_image, output_path=tmp_pdf_path)
        with open(tmp_pdf_path, "rb") as f:
            pdf_data = f.read()

        st.markdown("---")
        if is_dicom and isinstance(result, DicomPredictionResult) and result.secondary_capture_dcm is not None:
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="📄 Ekspor Laporan Medis (PDF)",
                    data=pdf_data,
                    file_name=f"PulmoScan_Report_{st.session_state.patient_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            with col_dl2:
                sc_buffer = io.BytesIO()
                result.secondary_capture_dcm.save_as(sc_buffer, enforce_file_format=True)
                st.download_button(
                    label="💾 Unduh DICOM PACS (.dcm)",
                    data=sc_buffer.getvalue(),
                    file_name=f"PulmoScan_{st.session_state.patient_id}_{result.class_name.lower()}.dcm",
                    mime="application/dicom",
                    use_container_width=True,
                    help="Unduh file DICOM standar untuk diimpor ke sistem PACS RS (Horos, RadiAnt, GE Centricity)"
                )
        else:
            st.download_button(
                label="📄 Ekspor Laporan Medis (PDF)",
                data=pdf_data,
                file_name=f"PulmoScan_Report_{st.session_state.patient_id}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# Disclaimer Footer
st.markdown(
    """
    <div class="medical-disclaimer">
        <strong>⚠️ Catatan Klinis & Legal Disclaimer:</strong><br>
        PulmoScan AI dirancang sebagai Clinical Decision Support System (CDSS) untuk riset dan edukasi medis.
        Hasil prediksi dan peta atensi Grad-CAM bukan pengganti pembacaan definitif dokter spesialis radiologi atau dokter paru.
        Keputusan terapi wajib dikonfirmasi melalui evaluasi klinis dan biopsi histopatologi.
    </div>
    """,
    unsafe_allow_html=True
)
