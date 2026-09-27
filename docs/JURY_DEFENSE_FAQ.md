# FAQ & Panduan Antisipasi Pertanyaan Juri (Defense Guide)
## PulmoScan AI: Intelligent Lung Cancer CT Detection & Explainability

Dokumen ini disusun sebagai panduan strategis dan pegangan cepat (*cheat sheet*) untuk menjawab pertanyaan teknis, metodologis, maupun etika klinis saat presentasi di depan dewan juri kompetisi (SIC / Hackathon / Portofolio Review) atau dokter spesialis.

---

### Q1: "Kenapa akurasi modelnya sangat tinggi (98%+)? Apakah tidak overfitting?"
**Jawaban Defensif:**
> *"Tingginya akurasi telah kami uji secara ketat melalui **5-Fold Group-Stratified Cross-Validation**, yang menghasilkan skor **98.59% ± 0.20%**. Standar deviasi yang sangat sempit ($\pm 0.20\%$) membuktikan bahwa model tidak overfitting pada satu partisi data tertentu. Selain itu, kami menerapkan regularisasi dropout ($p=0.2$), augmentasi rotasi dan jitter saat training, serta menggunakan arsitektur compact EfficientNet-B0 (hanya 4 juta parameter) untuk mencegah parameter memorization."*

---

### Q2: "Banyak paper AI paru terkena data leakage. Bagaimana kalian menjamin tidak ada kebocoran data?"
**Jawaban Defensif:**
> *"Dataset IQ-OTHNCCD di Kaggle memiliki kelemahan mendasar: ~1.000 citra asli di-augment secara offline menjadi 12.184 file dengan pola penamaan `{Class} case ({patient_id})({aug_id}).jpg`. Jika di-split per-file biasa, salinan pasien yang sama akan bocor ke data test.
>
> Kami memitigasi ini dengan membangun **Group-Aware Splitting** (`src/preprocessing/split.py`). Seluruh varian pasien dikunci ke satu partisi saja (70% train, 15% val, 15% test). Kami juga menyertakan unit test otomatis (`test_real_split_no_leakage`) yang memverifikasi tidak ada satu pun patient ID yang beririsan antar file split di disk."*

---

### Q3: "Kenapa memilih EfficientNet-B0, bukan Vision Transformer (ViT) atau ResNet-50?"
**Jawaban Defensif:**
> *"Kami memprioritaskan efisiensi komputasi dan implementasi nyata di fasilitas kesehatan daerah (Puskesmas/RSUD) yang sering kali tidak memiliki server GPU mahal.
> - **ResNet-50**: Memiliki ~23,5 juta parameter dengan latensi CPU ~400 ms.
> - **Vision Transformer (ViT)**: Membutuhkan data latih jutaan citra agar tidak overfitting.
> - **EfficientNet-B0**: Hanya memiliki **4,01 juta parameter (ukuran file model hanya 16 MB)**, tetapi menghasilkan akurasi yang lebih unggul (98.47%) dengan latensi inferensi super cepat **~150 ms di CPU biasa**. Ini membuat PulmoScan sangat realistis untuk di-deploy secara on-premise."*

---

### Q4: "Bagaimana cara kerja Explainable AI (Grad-CAM) di proyek ini?"
**Jawaban Defensif:**
> *"Radiolog tidak bisa menerima sistem black-box. Grad-CAM (Gradient-weighted Class Activation Mapping) bekerja dengan menghitung gradien parsial skor kelas target ($\frac{\partial y^c}{\partial A^k}$) terhadap activation map konvolusi terakhir (`features[-1]`). 
> 
> Nilai aktivasi positif kemudian diagregasi menggunakan fungsi ReLU dan di-overlay ke citra CT asli dengan alpha blending. Ini memberikan peta panas spasial (heatmap) yang membuktikan model mengambil keputusan berdasarkan tepi spikulasi nodul, bukan artefak teks atau background rongga dada."*

---

### Q5: "Di rumah sakit, data formatnya DICOM 16-bit, bukan JPG. Apakah sistem ini bisa dipakai?"
**Jawaban Defensif:**
> *"Bisa. Kami telah mengimplementasikan modul **Clinical DICOM Pipeline** (`src/dicom/processor.py`). Sistem kami:
> 1. Menerima file medis asli `.dcm` 16-bit.
> 2. Mengonversi nilai sensor ke skala radiasi standar **Hounsfield Units (HU)**: $\text{HU} = \text{Pixel} \times \text{Slope} + \text{Intercept}$.
> 3. Menerapkan **Standard Lung Windowing** ($W=1500, L=-600 \text{ HU}$) untuk memunculkan parenkim paru.
> 4. Mengekspor kembali hasil deteksi dan heatmap ke format **DICOM Secondary Capture (`.dcm`)**, sehingga dokter dapat membukanya langsung di workstation PACS rumah sakit seperti Horos atau RadiAnt."*

---

### Q6: "Apakah AI ini siap 100% menggantikan diagnosis dokter?"
**Jawaban Defensif (Wajib disampaikan secara etis):**
> *"Sama sekali tidak. PulmoScan dirancang sebagai **Clinical Decision Support System (CDSS) / Computer-Aided Triage Tool**, bukan pengganti dokter. 
> 
> Perannya adalah membantu menyaring antrean (*triage prioritization*): memprioritaskan pasien dengan indikasi nodul mencurigakan agar dibaca lebih awal oleh dokter spesialis. Penegakan diagnosis definitif tetap mutlak berada di tangan dokter spesialis paru dan radiolog melalui konfirmasi histopatologi dan biopsi klinis."*

---

### Ringkasan 3 Poin Kunci untuk Penutup Presentasi:
1. **Clinical Impact**: Menjembatani kesenjangan jumlah radiolog di Indonesia (~1,2 radiolog per 100.000 jiwa) dengan sistem triase cepat.
2. **Methodological Rigor**: Mencegah data leakage melalui Group-Aware Splitting dan divalidasi dengan 5-Fold Group Cross-Validation.
3. **PACS & DICOM Ready**: Kompatibel dengan alur kerja workstation radiologi rumah sakit nyata.
