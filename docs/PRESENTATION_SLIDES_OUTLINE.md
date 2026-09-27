# Outline Slide Presentasi — PulmoScan AI (Kompetisi / Pitch Deck)

Struktur presentasi 10 slide dirancang untuk juri teknis dan non-teknis (kombinasi kedalaman metodologi medis + keunggulan rekayasa AI).

---

### Slide 1: Judul & Hook (The Opening)
- **Judul:** PulmoScan AI: Intelligent Lung CT Cancer Detection & Explainability
- **Sub-judul:** Deteksi Dini Nodul Kanker Paru Berbasis Hybrid Deep Learning & Transparansi Klinis Grad-CAM
- **Presenter:** Suzirz
- **Visual:** Mockup antarmuka Streamlit berdampingan dengan peta atensi Grad-CAM.

---

### Slide 2: Urgensi Masalah (The Clinical Problem)
- **Fakta Global:** Kanker paru adalah penyebab kematian akibat kanker nomor 1 di dunia (WHO: ~1,8 juta kematian per tahun).
- **Bottleneck Deteksi Dini:** Skrining CT scan memerlukan pembacaan manual ratusan potongan aksial per pasien oleh radiolog spesialis.
- **Tantangan di Indonesia & Negara Berkembang:** Rasio radiolog spesialis paru sangat terbatas; risiko *fatigue-induced misdiagnosis* dan *false negatives* yang berakibat fatal.

---

### Slide 3: Solusi Kami (The Proposed Solution)
- **PulmoScan AI:** Sistem pendukung keputusan klinis (*Clinical Decision Support System / CDSS*) berbasis web.
- **3 Kemampuan Utama:**
  1. Klasifikasi akurat 3 kelas: *Normal*, *Benign (Jinak)*, dan *Malignant (Ganas)*.
  2. Lapisan *Explainability* (Grad-CAM): Bukan sekadar black-box — radiolog dapat melihat area nodul yang mendasari keputusan AI.
  3. Responsivitas instan (~150 ms) tanpa infrastruktur server berat.

---

### Slide 4: Dataset & Metodologi Group-Aware (Anti-Leakage)
- **Dataset:** IQ-OTH/NCCD Lung Cancer Dataset (12.184 citra CT scan terlabel klinis).
- **Group-Aware Splitting (Patient Isolation):** Memitigasi offline augmentation leakage dengan mengisolasi seluruh varian scan pasien yang sama ke satu partisi saja (70% Train, 15% Val, 15% Test).
- **Integritas Medis:** Didukung unit test otomatis (`test_real_split_no_leakage`) yang memvalidasi ketiadaan tumpang-tindih ID pasien antar split.
- **Augmentasi Runtime Medis:** Random horizontal/vertical flips, rotasi (±15°), dan normalisasi ruang warna ImageNet.

---

### Slide 5: Arsitektur Sistem (Multimodal Hybrid Pipeline)
- **Diagram Alir:**
  1. Input CT Scan aksial (224x224 px) atau Berkas Medis Asli DICOM 16-bit
  2. CNN Backbone: Pretrained **EfficientNet-B0** (Feature Extractor → 1.280 dimensi embedding)
  3. Cabang Radiomik: 13 Fitur Kuantitatif (Intensity Moments, Shannon Entropy, Laplacian Variance)
  4. Evaluasi Klasifikasi:
     - End-to-End CNN Classification Head (Dropout + Linear)
     - Hybrid Classical ML: Random Forest (Balanced Class Weights) & XGBoost
  5. Explainability Layer: Grad-CAM pada convolutional layer terakhir (`features[-1]`).

---

### Slide 6: Pelatihan & Rekayasa Hardware (Google Colab Tesla T4)
- **Hardware Target:** NVIDIA Tesla T4 GPU (16 GB VRAM).
- **Optimalisasi:** PyTorch Automatic Mixed Precision (**FP16 AMP**) untuk efisiensi komputasi dan memori.
- **Konvergensi:** Checkpoint dipandu oleh metrik *Validation Recall* untuk memprioritaskan zero false negative pada kasus kanker ganas.

---

### Slide 7: Hasil Evaluasi & Validasi Medis (Audit Realistis)
- **Tabel Metrik Utama (Group-Aware Held-Out Test):**
  - **Malignant Recall / Sensitivity:** **100.0%** (624 dari 624 scan ganas terdeteksi; 0 False Negative)
  - **Overall Test Accuracy:** **98.47%** (1.733 dari 1.760 scan terklasifikasi benar)
  - **AUC-ROC (Multi-Class OvR):** **0.999** (Pemisahan probabilitas sangat tajam)
  - **5-Fold Group Cross-Validation:** **98.59% ± 0.20%** (Variansi sempit membuktikan kestabilan model lintas pasien)
- **Analisis Kesalahan:** 26 kasus normal terprediksi benign akibat densitas bronkovaskular/hilar, 1 benign terprediksi malignant akibat batas lesi ireguler; **0 kasus malignant yang terlewat**.

---

### Slide 8: Explainable AI (Grad-CAM in Action)
- **Mengapa Explainability Wajib di Medis?** Dokter tidak akan mempercayai rekomendasi AI tanpa transparansi spasial.
- **Studi Kasus:**
  - Kasus Malignant: Heatmap terkonsentrasi tepat pada densitas massa berbatas spikulasi.
  - Kasus Benign: Heatmap menunjukkan lesi noduler bulat berbatas tegas.
  - Kasus Normal: Tidak ada konsentrasi atensi anomali pada parenkim paru.

---

### Slide 9: Demo Produk & Integrasi Rumah Sakit (PACS Ready)
- **Live Demo Streamlit:** Menampilkan antarmuka Dark Medical Slate.
- **Fitur Unggulan Klinis:**
  - Pengujian instan dengan contoh klinis bawaan (JPG maupun berkas DICOM asli).
  - Pembacaan atenuasi **Hounsfield Units (HU)** dan kalibrasi *Standard Lung Window* (-600 HU / 1500 W).
  - **Ekspor DICOM Secondary Capture (`.dcm`)**: Hasil analisis dan heatmap Grad-CAM dapat langsung diunduh dan dibuka di PACS workstation radiolog (Horos, RadiAnt, GE Centricity).

---

### Slide 10: Rencana Pengembangan & Dampak Klinis (Future Roadmap)
- **Tahap Selanjutnya:**
  1. Validasi eksternal multi-senter pada repositori publik LIDC-IDRI (1.018 pasien, 7 institusi).
  2. Implementasi background PACS listener otomatis (DICOM C-STORE daemon).
  3. Pengajuan sertifikasi Software as a Medical Device (SaMD) dan uji klinis pendamping radiolog.
- **Penutup & Tanya Jawab:** "Empowering Radiologists with Explainable Intelligence."
