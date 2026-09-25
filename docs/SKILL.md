# SKILL.md — Domain Knowledge yang Dibutuhkan

Catatan skill/knowledge yang perlu dikuasai atau dipelajari buat ngerjain project ini dengan bener.

## 1. Medical Imaging Basics
* **CT Scan & Hounsfield Unit (HU)** — satuan densitas jaringan di CT scan; paru punya rentang HU khas (windowing perlu disesuaikan)
* **Nodul paru** — massa kecil di jaringan paru; bisa benign (jinak) atau malignant (ganas), ukuran & bentuk jadi indikator penting
* **DICOM format** — format standar file medical imaging, beda dari JPG/PNG biasa (pakai `pydicom` buat baca)

## 2. Deep Learning
* **Transfer Learning** — pakai model pretrained (ImageNet: ResNet, EfficientNet) lalu fine-tune ke domain medis → efektif buat dataset kecil
* **CNN Feature Extraction** — ambil output dari layer sebelum fully-connected sebagai representasi fitur citra
* **Data Augmentation** — penting banget di medical imaging karena dataset biasanya kecil (rotate, flip, zoom, brightness)
* **Grad-CAM** — teknik visualisasi buat lihat area citra mana yang paling berpengaruh ke keputusan CNN

## 3. Machine Learning Klasik
* **Random Forest / XGBoost** — classifier robust buat data tabular/fitur hasil ekstraksi, tahan overfitting di data kecil
* **Radiomics features** — fitur kuantitatif dari citra medis: shape (bentuk), texture (GLCM, GLRLM), intensity (histogram) — diekstrak pakai `pyradiomics`
* **Class imbalance handling** — SMOTE, class weighting (penting karena data malignant biasanya lebih sedikit dari benign/normal)

## 4. Evaluasi Model Medis
* **Kenapa recall > accuracy** — false negative (bilang sehat padahal kanker) jauh lebih berbahaya daripada false positive
* **Confusion Matrix, ROC-AUC** — wajib dipahami buat presentasi hasil ke juri
* **Cross-validation** — penting di dataset kecil biar hasil evaluasi ga bias

## 5. Tools yang Perlu Dipelajari
* `pydicom` / `SimpleITK` — baca & proses file CT scan
* `pyradiomics` — ekstraksi fitur radiomics
* `TensorFlow/Keras` atau `PyTorch` — training CNN
* `scikit-learn`, `xgboost` — training classifier
* `Streamlit`/`Gradio` — bikin demo cepat tanpa ribet frontend

## 6. Pelajaran Praktis & Gotchas dari Proyek Ini
1. **Cross-Platform Path Hygiene (Windows vs Linux Colab)**:
   - Metadata JSON dataset yang dihasilkan di Windows menggunakan backslash (`\`) akan gagal dibaca di Linux Google Colab.
   - Solusi: Selalu lakukan normalisasi `.replace("\\", "/")` di DataLoader dan generator split.
2. **PyTorch Mixed Precision (AMP)**:
   - Penggunaan `torch.cuda.amp.autocast()` dan `GradScaler()` memangkas penggunaan VRAM hingga ~50% dan mempercepat training ~2x di Tesla T4 Tensor Cores tanpa degradasi akurasi.
3. **Optimasi Metrik Medis vs Kerugian Medis (False Negative Fatal)**:
   - Akurasi tinggi tidak ada artinya jika kasus *Malignant* salah diprediksi sebagai *Normal* (*False Negative*).
   - Checkpointing model wajib dipandu oleh *Validation Recall*, bukan sekadar *Validation Loss*.
4. **Grad-CAM Hook Lifecycle**:
   - `input_tensor.requires_grad = True` wajib diaktifkan saat inferensi Grad-CAM meskipun model berada di mode `eval()`, agar graf komputasi diferensiasi balik dapat menghitung gradien saluran fitur konvolusi.
5. **Matplotlib Headless Server Rendering**:
   - Menghasilkan plot pada pengujian otomatis headless wajib menyertakan `plt.close(fig)` untuk mencegah kebocoran memori GDI di Windows atau OOM di server.

## 7. Tips Menjawab Pertanyaan Juri Kompetisi (Contest Defense Q&A)
- **Q: Kenapa memilih EfficientNet-B0 daripada ResNet50 atau Vision Transformer?**
  - *A: EfficientNet-B0 memanfaatkan Compound Scaling dengan efisiensi parameter sangat tinggi (~4 juta parameter vs 25 juta pada ResNet50). Model ini sangat cepat konvergen, hemat komputasi (16.3 MB), dan cocok untuk inferensi edge klinis atau cloud tanpa latensi tinggi.*
- **Q: Kenapa perlu pendekatan Hybrid (CNN + Random Forest / XGBoost)?**
  - *A: CNN bertindak sebagai representational feature learner yang memetakan citra CT beresolusi tinggi ke manifold 1.280 dimensi. Random Forest dan XGBoost menyediakan batasan keputusan ensemble dengan pembobotan kelas seimbang (`class_weight='balanced'`), memberikan stabilitas tambahan dan resistensi tinggi terhadap outlier.*
- **Q: Mengapa dokter harus mempercayai sistem ini?**
  - *A: Sistem ini dirancang sebagai Clinical Decision Support System (CDSS) yang explainable melalui Grad-CAM. Dokter dapat melihat secara visual apakah perhatian model benar-benar terfokus pada spikulasi nodul paru atau hanya artefak latar belakang CT scan.*
