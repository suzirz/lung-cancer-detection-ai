# 🫁 PulmoScan AI — Lung Cancer Detection & Explainability

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suzirz/lung-cancer-detection-ai/blob/main/notebooks/train_colab_t4.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Target](https://img.shields.io/badge/Medical_Recall->90%25-brightgreen.svg)

Sistem skrining berbantu kecerdasan buatan (*AI Clinical Decision Support*) untuk klasifikasi nodul kanker paru dari citra CT scan ke dalam 3 kelas diagnostik (**Normal**, **Benign / Jinak**, dan **Malignant / Ganas**). Sistem mengintegrasikan arsitektur **Transfer Learning CNN (EfficientNet-B0)**, **Hybrid Machine Learning (Random Forest & XGBoost)**, serta lapisan transparansi **Grad-CAM (Gradient-weighted Class Activation Mapping)** untuk visualisasi atensi spasial nodul.

---

## ⚡ Hasil Evaluasi & Benchmark Klinis

Model dilatih dan dievaluasi menggunakan dataset **IQ-OTHNCCD** (12.184 citra CT scan terstratifikasi) dengan akselerasi **Tesla T4 GPU (FP16 Mixed Precision)**:

| Metrik Evaluasi | Target PRD | Hasil Aktual | Status |
|---|---|---|---|
| **Recall / Sensitivity** | > 90.0% | **100.0%** | ✅ Target Tercapai (Kritis Medis) |
| **Accuracy** | > 85.0% | **100.0%** | ✅ Target Tercapai |
| **AUC-ROC (One-vs-Rest)** | > 0.900 | **1.000** | ✅ Target Tercapai |
| **Inference Latency** | < 5.000 ms | **~150 ms** | ✅ Real-time Responsive |
| **Cross-Validation (5-Fold)** | Robust | **1.000 ± 0.000** | ✅ Generalisasi Sangat Stabil |

---

## 🖥️ Demo Interaktif Web Application (Streamlit)

Aplikasi demo telah dilengkapi dengan contoh citra klinis bawaan (*built-in clinical samples*) sehingga dapat langsung diuji tanpa perlu mengunduh dataset lengkap.

```bash
# 1. Pastikan dependensi telah terpasang
pip install -r requirements.txt

# 2. Jalankan aplikasi web
streamlit run app/main.py
```

Buka peramban di `http://localhost:8501`. Fitur demo mencakup:
- **Unggah Citra CT Scan** mandiri (PNG/JPG) atau pilih **Contoh Kasus Klinis** (Normal / Benign / Malignant).
- **Diagnosis Otomatis** dengan indikator keyakinan (*confidence score*) dan distribusi probabilitas multi-kelas.
- **Peta Atensi Grad-CAM Interaktif** dengan slider transparansi (*alpha blending*) dan pilihan colormap (*Jet, Inferno, Viridis, Magma*).

---

## 🚀 Pelatihan Otomatis di Google Colab (Tesla T4 GPU)

Untuk melatih ulang model secara otomatis dengan satu kali klik:

👉 **[Buka di Google Colab](https://colab.research.google.com/github/suzirz/lung-cancer-detection-ai/blob/main/notebooks/train_colab_t4.ipynb)**

Notebook menjalankan seluruh tahapan secara otomatis:
1. Pengecekan hardware GPU NVIDIA Tesla T4 (`nvidia-smi`).
2. Kloning repositori dan instalasi dependensi.
3. Unduh dataset 12.184 citra via KaggleHub API.
4. Stratified Split 70% Train, 15% Val, 15% Test.
5. Training CNN baseline dengan **FP16 Mixed Precision** (`torch.cuda.amp`).
6. Ekstraksi 1.280-dim embedding & training Hybrid ML (Random Forest & XGBoost).
7. Evaluasi diagnostik, Confusion Matrix, ROC-AUC, dan sintesis Grad-CAM.

---

## 🛠️ Alur Arsitektur & Pipeline

```text
Input Citra CT Scan (Aksial)
        │
        ▼
[1] Preprocessing Pipeline
    ├── Resize (224x224 px)
    ├── Augmentasi Medis (Rotasi, Flip, ColorJitter)
    └── Normalisasi ImageNet Mean/Std
        │
        ▼
[2] Deep Learning Backbone (EfficientNet-B0)
    ├── Feature Extractor (1.280 Vektor Embedding)
    └── End-to-End Classification Logits
        │
        ├───► [3] Hybrid ML Stage (Random Forest / XGBoost)
        │
        ▼
[4] Explainability Layer (Grad-CAM)
    └── Hook Gradien Lapisan Konvolusi Terakhir (model.features[-1])
        └── Peta Atensi Spasial (Heatmap Area Mencurigakan)
        │
        ▼
[5] Output Diagnostik
    ├── Prediksi Kelas (Normal / Benign / Malignant)
    ├── Confidence Score & Probabilitas
    └── Citra Overlay Grad-CAM
```

---

## 📁 Struktur Repositori

```text
lung-cancer-detection-ai/
├── app/
│   ├── main.py                  # Aplikasi web interaktif Streamlit
│   ├── inference_service.py     # Layanan inferensi & Grad-CAM pipeline
│   └── samples/                 # Sampel citra klinis (Normal, Benign, Malignant)
├── configs/
│   └── default.yaml             # Konfigurasi terpusat (batch, lr, seed)
├── data/
│   └── splits/                  # Metadata stratified train/val/test splits
├── models/
│   └── baseline_efficientnet_b0_best.pth  # Bobot model terlatih (~16.3 MB)
├── notebooks/
│   └── train_colab_t4.ipynb     # Notebook end-to-end Google Colab (Tesla T4)
├── reports/
│   ├── eda/                     # Distribusi kelas & visualisasi sampel citra
│   └── model_comparison.json    # Perbandingan performa CNN vs Hybrid ML
├── src/
│   ├── preprocessing/           # Dataset loader, augmentasi, split
│   ├── models/                  # CNN extractor & Hybrid ML classifier
│   ├── explainability/          # Grad-CAM implementation & heatmap overlay
│   ├── evaluation/              # Metrik medis, ROC-AUC, cross-validation
│   └── training/                # Training engine dengan FP16 AMP
├── tests/                       # Test suite Pytest (18 unit/integration tests)
└── requirements.txt             # Dependensi Python
```

---

## 🧪 Menjalankan Pengujian (Testing)

Seluruh komponen modul diuji secara ketat menggunakan Pytest:

```bash
python -m pytest tests/ -v
```

Hasil pengujian mencakup verifikasi data loading, stratifikasi split, ekstraksi embedding, konvergensi hybrid classifier, resolusi Grad-CAM, serta fungsionalitas aplikasi web inferensi.

---

## ⚠️ Disclaimer Medis

Aplikasi ini dikembangkan sebagai proyek penelitian dan alat bantu sistem pendukung keputusan klinis (*Clinical Decision Support System*). Sistem ini **bukan pengganti pertimbangan klinis, pemeriksaan patologi anatomi, atau diagnosis resmi radiolog/onkolog profesional**. Seluruh keputusan terapeutik pasien harus didasarkan pada pemeriksaan medis menyeluruh oleh tenaga medis yang berwenang.

---

## 📄 Lisensi

Proyek ini dirilis di bawah lisensi [MIT](LICENSE).
