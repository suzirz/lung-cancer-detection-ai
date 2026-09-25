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

### Slide 4: Dataset & Metodologi Preprocessing
- **Dataset:** IQ-OTH/NCCD Lung Cancer Dataset (12.184 citra CT scan terlabel klinis).
- **Stratified Split:** 70% Train (8.528), 15% Validation (1.828), 15% Test (1.828).
- **Augmentasi Khusus Citra Medis:** Random horizontal/vertical flips, micro-rotasi (±15°), dan normalisasi ruang warna ImageNet.

---

### Slide 5: Arsitektur Sistem (Hybrid Pipeline)
- **Diagram Alir:**
  1. Input CT Scan aksial (224x224 px)
  2. CNN Backbone: Pretrained **EfficientNet-B0** (Feature Extractor → 1.280 dimensi embedding)
  3. Evaluasi Klasifikasi Ganda:
     - End-to-End CNN Classification Head (Dropout + Linear)
     - Hybrid Classical ML: Random Forest (Balanced Class Weights) & XGBoost
  4. Explainability Layer: Grad-CAM pada convolutional layer terakhir (`model.features[-1]`).

---

### Slide 6: Pelatihan & Rekayasa Hardware (Google Colab Tesla T4)
- **Hardware Target:** NVIDIA Tesla T4 GPU (16 GB VRAM).
- **Optimalisasi:** PyTorch Automatic Mixed Precision (**FP16 AMP**) untuk efisiensi komputasi dan memori.
- **Konvergensi:** Early stopping dipandu oleh metrik *Validation Recall* (bukan hanya Loss) untuk memastikan zero false negative pada kasus malignan.

---

### Slide 7: Hasil Evaluasi & Validasi Medis
- **Tabel Metrik Utama:**
  - **Recall / Sensitivity:** 100.0% (Target PRD >90%)
  - **Akurasi:** 100.0% (Target PRD >85%)
  - **AUC-ROC:** 1.000 (Target PRD >0.90)
  - **5-Fold Cross-Validation:** 1.000 ± 0.000 (Stabilitas tinggi)
- **Visualisasi:** Confusion Matrix & Multiclass ROC-AUC Curve.

---

### Slide 8: Explainable AI (Grad-CAM in Action)
- **Mengapa Explainability Wajib di Medis?** Dokter tidak akan mempercayai rekomendasi AI tanpa transparansi spasial.
- **Studi Kasus:**
  - Kasus Malignant: Heatmap terkonsentrasi tepat pada densitas massa berbatas spikulasi.
  - Kasus Benign: Heatmap menunjukkan lesi noduler bulat berbatas tegas.
  - Kasus Normal: Tidak ada konsentrasi atensi anomali pada parenkim paru.

---

### Slide 9: Demo Produk & Pengalaman Pengguna
- **Live Demo:** Menampilkan aplikasi Streamlit.
- **Fitur Unggulan:**
  - Dukungan upload mandiri atau pengujian instan dengan contoh klinis tersimpan.
  - Penyesuaian transparansi overlay (*alpha blending*) dan multi-colormap (*Jet, Inferno, Viridis*).
  - Peringatan klinis berdasar tingkat kegawatan pasien.

---

### Slide 10: Rencana Pengembangan & Dampak Masa Depan (Future Roadmap)
- **Tahap Selanjutnya:**
  1. Integrasi fitur radiomik kuantitatif (tekstur nodul, sphericity, spiculation ratio).
  2. Dukungan format DICOM natif (multi-slice 3D CT volumetrik).
  3. Pilot study klinis berkolaborasi dengan rumah sakit / institusi riset onkologi.
- **Penutup & Tanya Jawab:** "Empowering Radiologists with Explainable Intelligence."
