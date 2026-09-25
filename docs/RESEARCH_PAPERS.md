# Ringkasan Paper Referensi: Lung Cancer Classification & Radiomics

Dokumen ini merangkum 3 paper referensi utama sebagai landasan metodologi proyek `lung-cancer-detection-ai`.

---

## 1. Paper 1: Transfer Learning Benchmark pada IQ-OTH/NCCD Dataset
- **Judul**: *Classification of Lung Cancer from CT-Scan Images Using Transfer Learning*
- **Arsitektur yang Dievaluasi**: ResNet50, EfficientNet, VGG16, InceptionV3.
- **Temuan Kunci**:
  - Pretrained model pada ImageNet mampu beradaptasi cepat dengan bobot awal transfer learning pada citra CT scan paru.
  - EfficientNet menghasilkan rasio akurasi terhadap jumlah parameter (efisiensi komputasi) terbaik.
  - Tantangan utama: Citra non-nodul (dinding dada, tulang rusuk, diafragma) sering mengacaukan prediksi jika tidak dinormalisasi.
- **Relevansi ke Proyek Kita**:
  - Mengonfirmasi pemilihan **EfficientNet-B0** dan **ResNet50** sebagai backbone feature extractor baseline.
  - Normalisasi ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) adalah standar efektif.

---

## 2. Paper 2: Hybrid Deep Learning & Classical Machine Learning
- **Judul**: *Lung Cancer Classification Based on CT Images Using Hybrid Convolutional Neural Network - Random Forest*
- **Metode**:
  - Lapisan konvolusi CNN digunakan murni untuk mengekstrak vektor fitur (embedding) berdimensi tinggi (512–1280 dimensi).
  - Klasifikasi akhir digantikan oleh Random Forest / XGBoost classifier alih-alih lapisan Dense/Softmax standar.
- **Temuan Kunci**:
  - Classifier berbasis pohon keputusan (Ensemble Trees: Random Forest/XGBoost) lebih tahan terhadap overfitting pada variasi citra medis dibanding single Softmax layer.
  - Membantu menangani ketidakseimbangan kelas (*class weighting* bawaan Random Forest).
  - Skor **Recall pada kasus ganas (Malignant)** meningkat rata-rata 3–5% dibanding CNN murni.
- **Relevansi ke Proyek Kita**:
  - Menjadi justifikasi akademis kuat untuk arsitektur **Hybrid (CNN + Random Forest/XGBoost)** yang kita targetkan di [PRD.md](file:///c:/Users/Administrator/Documents/Project/lung-cancer-detection-ai/docs/PRD.md) dan [ARCHITECTURE.md](file:///c:/Users/Administrator/Documents/Project/lung-cancer-detection-ai/docs/ARCHITECTURE.md).

---

## 3. Paper 3: Radiomics & Explainability (Grad-CAM)
- **Judul**: *Explainable AI for Lung Nodule Malignancy Prediction with Deep Features and Radiomics*
- **Metode**:
  - Pemanfaatan **Grad-CAM (Gradient-weighted Class Activation Mapping)** untuk memvalidasi fokus spasial konvolusi terakhir.
  - Ekstraksi fitur radiomics (tekstur GLCM, bentuk/sphericity, intensitas) untuk melengkapi fitur abstrak CNN.
- **Temuan Kunci**:
  - Dokter radiolog dan juri kompetisi tidak bisa menerima model sebagai "black-box".
  - Grad-CAM membuktikan apakah model benar-benar melihat lesi nodul paru atau hanya melihat artefak tepi citra.
- **Relevansi ke Proyek Kita**:
  - Menetapkan target modul explainability di Minggu 4 agar langsung menghasilkan visualisasi heatmap aktivasi nodul pada demo Streamlit.
