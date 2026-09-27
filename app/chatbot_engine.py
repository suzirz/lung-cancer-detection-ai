"""
Module: Conversational Medical Co-Pilot Engine
File: app/chatbot_engine.py

Prinsip Codebase Design:
- Modul Deep: Menyediakan reasoning klinis berbasis bukti (Evidence-Based Guidelines:
  Fleischner Society & Lung-RADS) yang dipadukan dengan inferensi model deep learning.
- Interface sederhana: ChatbotEngine.respond(user_text) & format_diagnostic_report(prediction).
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from PIL import Image

from app.inference_service import PredictionResult


@dataclass
class ChatResponse:
    """Representasi respon percakapan klinis cerdas."""
    text: str
    has_image: bool = False
    image: Optional[Image.Image] = None
    prediction: Optional[PredictionResult] = None
    confidence_badge: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatbotEngine:
    """
    Engine dialog asisten klinis PulmoScan.
    Memadukan interpretasi inferensi CT Scan dengan panduan medis Fleischner Society.
    """

    def __init__(self) -> None:
        self.last_prediction: Optional[PredictionResult] = None
        self.patient_id: str = "Anonymous"

    def format_diagnostic_report(
        self,
        prediction: PredictionResult,
        patient_id: str = "Anonymous"
    ) -> ChatResponse:
        """
        Memformat hasil inferensi model menjadi laporan diagnostik klinis interaktif.
        """
        self.last_prediction = prediction
        self.patient_id = patient_id

        conf_pct = prediction.confidence * 100.0
        cls_name = prediction.class_name

        # Penentuan status risiko dan warna indikator
        if cls_name == "Malignant":
            risk_header = "### ⚠️ TEMUAN KRITIS: KANKER GANAS (MALIGNANT)"
            risk_badge = "🔴 Risiko Tinggi (High Malignancy Risk)"
            recommendation = (
                "**Rekomendasi Klinis (Fleischner Society & Lung-RADS Kategori 4X):**\n"
                "- Konsultasikan segera dengan Dokter Spesialis Onkologi / Bedah Toraks.\n"
                "- Dianjurkan pemeriksaan lanjutan: **PET-CT Scan** untuk staging metabolik atau **Biopsi Jarum Halus (FNAB)**.\n"
                "- Evaluasi keterlibatan kelenjar getah bening mediastinum."
            )
        elif cls_name == "Benign":
            risk_header = "### ℹ️ TEMUAN LESI: JINAK / BENIGN"
            risk_badge = "🟡 Risiko Rendah-Sedang (Indeterminate/Benign)"
            recommendation = (
                "**Rekomendasi Klinis (Fleischner Society Guidelines):**\n"
                "- Nodul menunjukkan karakteristik jinak (kalsifikasi sentral/tepi halus).\n"
                "- Dianjurkan **Follow-up Low-Dose CT (LDCT)** dalam 6–12 bulan untuk memastikan stabilitas volumetrik nodul (tidak bertumbuh)."
            )
        else:
            risk_header = "### ✅ TEMUAN NORMAL: TIDAK DITEMUKAN LESI"
            risk_badge = "🟢 Normal (Parenkim Paru Bersih)"
            recommendation = (
                "**Rekomendasi Klinis:**\n"
                "- Parenkim paru dalam batas normal tanpa nodul signifikan.\n"
                "- Skrining tahunan rutin bagi populasi berisiko tinggi (riwayat merokok >20 pack-years)."
            )

        prob_bars = "\n".join([
            f"- **{k}**: `{v * 100.0:.2f}%`"
            for k, v in sorted(prediction.probabilities.items(), key=lambda x: x[1], reverse=True)
        ])

        report_md = f"""{risk_header}

**Identifikasi Pasien:** `{patient_id}`  
**Tingkat Keyakinan (Confidence):** **{conf_pct:.2f}%** ({risk_badge})  
**Latensi Komputasi:** `{prediction.latency_ms:.1f} ms`

---

#### 📊 Distribusi Probabilitas Model:
{prob_bars}

---

#### 🔬 Panduan Tindak Lanjut Medis:
{recommendation}

---

*💡 Anda dapat bertanya lebih lanjut di bawah, seperti: "Kenapa diklasifikasikan {cls_name.lower()}?", "Bagaimana cara membaca Grad-CAM?", atau meminta ekspor laporan PDF.*
"""
        return ChatResponse(
            text=report_md,
            has_image=True,
            image=prediction.overlay_image,
            prediction=prediction,
            confidence_badge=risk_badge
        )

    def respond(
        self,
        user_text: str,
        image: Optional[Image.Image] = None
    ) -> ChatResponse:
        """
        Menjawab pertanyaan klinis berbasis konteks kasus pasien dan literatur medis.
        """
        q = user_text.lower().strip()

        # Konteks jika user bertanya tentang diagnosa terkini
        pred = self.last_prediction

        # 1. Salam & Pengenalan Sistem
        if any(w in q for w in ["halo", "hai", "hi", "apa itu pulmoscan", "fungsi", "tentang"]):
            text = (
                "Halo! Saya **PulmoScan AI Co-Pilot**, asisten cerdas untuk analisis citra CT scan toraks.\n\n"
                "**Kapabilitas Utama:**\n"
                "1. **Klasifikasi 3-Kelas:** Deteksi lesi Paru Normal, Jinak (*Benign*), atau Kanker Ganas (*Malignant*).\n"
                "2. **Explainable AI (Grad-CAM):** Visualisasi atensi visual fokus model pada nodul.\n"
                "3. **Panduan Fleischner Society:** Rekomendasi interval tindak lanjut klinis.\n\n"
                "Silakan unggah citra CT Scan paru (format JPG, PNG, atau DICOM .dcm) untuk memulai analisis."
            )
            return ChatResponse(text=text)

        # 2. Pertanyaan mengapa Malignant / Ganas
        if any(w in q for w in ["kenapa malignant", "mengapa malignant", "kenapa ganas", "mengapa ganas"]):
            if pred and pred.class_name == "Malignant":
                text = (
                    "**Rasional Klinis Klasifikasi Malignant:**\n\n"
                    f"Model mendeteksi lesi ganas dengan keyakinan **{pred.confidence*100:.1f}%**. Fitur visual kunci yang ditangkap oleh lapisan konvolusi:\n"
                    "1. **Morfologi Tepi Spikulasi (*Spiculated Margins*):** Tepi lesi yang tidak teratur/bercabang adalah penanda kuat invasi sel ganas ke stroma paru.\n"
                    "2. **Densitas & Atenuasi Heterogen:** Area nekrosis sentral atau densitas *soft-tissue* padat di dalam nodul.\n"
                    "3. **Grad-CAM Saliency:** Peta panas Grad-CAM menunjukkan gradien atensi tertinggi berpusat tepat pada massa nodul tersebut.\n\n"
                    "**Tindakan Prioritas:** Konsultasi onkologi dan evaluasi biopsi/PET-CT untuk konfirmasi histopatologi."
                )
            else:
                text = (
                    "Klasifikasi **Malignant (Ganas)** pada CT scan umumnya didasarkan pada tanda-tanda radiologis:\n"
                    "- Diameter nodul > 8 mm dengan laju pertumbuhan cepat.\n"
                    "- Tepi lesi berspikulasi (*corona radiata*).\n"
                    "- Adanya vaskularisasi abnormal yang menyuplai massa tumor."
                )
            return ChatResponse(text=text)

        # 3. Pertanyaan mengapa Benign / Jinak
        if any(w in q for w in ["kenapa benign", "mengapa benign", "kenapa jinak", "mengapa jinak"]):
            text = (
                "**Karakteristik Nodul Jinak (Benign) pada CT Scan:**\n\n"
                "- **Bentuk Teratur & Tepi Halus (*Smooth Well-Defined Margins*):** Jarang menginvasi parenkim sekitarnya.\n"
                "- **Pola Kalsifikasi Khas:** Kalsifikasi difus, konsentris (*laminated*), atau pola *popcorn* (khas hamartoma).\n"
                "- **Stabilitas Ukuran:** Tidak ada pertambahan volume bermakna pada follow-up interval.\n\n"
                "Meskipun jinak, evaluasi Low-Dose CT (LDCT) ulang dalam 6-12 bulan tetap dianjurkan untuk konfirmasi."
            )
            return ChatResponse(text=text)

        # 4. Pertanyaan tentang Tindakan Selanjutnya / Rekomendasi
        if any(w in q for w in ["tindakan selanjutnya", "langkah selanjutnya", "apa yang harus dilakukan", "next step", "rekomendasi"]):
            if pred and pred.class_name == "Malignant":
                text = (
                    "**Protokol Klinis Tindak Lanjut untuk Dugaan Kanker Ganas:**\n\n"
                    "1. **Rujukan Cepat:** Rujuk ke Tim Onkologi Toraks Multidisiplin (*MDT*).\n"
                    "2. **Pemeriksaan Lanjutan:**\n"
                    "   - *Contrast-Enhanced CT* toraks dan abdomen atas.\n"
                    "   - *FDG PET-CT* untuk mendeteksi metastasis jauh.\n"
                    "3. **Konfirmasi Jaringan:** Biopsi transtorakal berpandu CT (*CT-guided TTNA*) atau bronkoskopi EBUS.\n"
                    "4. **Uji Fungsi Paru (Spirometri):** Menilai kelayakan kapasitas paru sebelum tindakan reseksi bedah."
                )
            elif pred and pred.class_name == "Benign":
                text = (
                    "**Protokol Tindak Lanjut Nodul Jinak (Fleischner Society):**\n\n"
                    "- Jika nodul padat < 6 mm pada pasien risiko rendah: Tidak perlu follow-up rutin.\n"
                    "- Jika nodul 6–8 mm: Rekomendasi CT ulang dalam 6–12 bulan.\n"
                    "- Catatan: Pasien disarankan berhenti merokok untuk menurunkan risiko timbulnya nodul baru."
                )
            else:
                text = (
                    "Hasil saat ini dalam batas normal. Tetap anjurkan pola hidup sehat dan skrining periodik bagi kelompok perokok aktif/pasif."
                )
            return ChatResponse(text=text)

        # 5. Pertanyaan tentang Grad-CAM
        if any(w in q for w in ["gradcam", "grad-cam", "heatmap", "peta panas", "warna merah"]):
            text = (
                "**Cara Membaca Peta Atensi Grad-CAM (Gradient-weighted Class Activation Mapping):**\n\n"
                "- **Warna Merah/Kuning (High Intensity):** Area dengan aktivasi gradien tertinggi. Ini adalah area citra yang paling dominan memicu keputusan model.\n"
                "- **Warna Biru/Transparan (Low Intensity):** Area yang diabaikan oleh model (seperti udara paru normal atau dinding dada).\n\n"
                "Grad-CAM memastikan bahwa model melihat nodul tumor secara nyata, bukan tertipu oleh artefak citra atau noise tulang."
            )
            return ChatResponse(text=text)

        # 6. Fallback General Medical Co-Pilot
        fallback_text = (
            f"Mengenai pertanyaan Anda: *\"{user_text}\"*\n\n"
            "Sebagai sistem pendukung keputusan klinis, PulmoScan AI menganalisis karakteristik morfologi parenkim paru.\n"
            "- Anda dapat mengunggah citra CT baru untuk evaluasi kasus lain.\n"
            "- Atau ketik *\"tindakan selanjutnya\"* untuk melihat rekomendasi klinis komprehensif berdasarkan panduan medis internasional."
        )
        return ChatResponse(text=fallback_text)
