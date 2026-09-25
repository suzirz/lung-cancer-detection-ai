# TODO — Lung Cancer Detection AI

## Minggu 1 — Riset & Setup
- [x] Download & eksplorasi dataset IQ-OTH/NCCD
- [x] Baca 2-3 paper referensi (nodule classification + radiomics)
- [x] Setup repo (struktur folder sesuai `ARCHITECTURE.md`)
- [x] Setup environment (`requirements.txt`, virtualenv)
- [x] EDA: distribusi kelas, ukuran citra, cek imbalance

## Minggu 2 — Preprocessing & Baseline CNN
- [x] Preprocessing pipeline (resize, normalisasi, windowing)
- [x] Split data (train/val/test)
- [x] Data augmentation
- [x] Training baseline CNN (transfer learning EfficientNet/ResNet) — TANPA hybrid dulu
- [x] Evaluasi baseline (accuracy, recall, confusion matrix)

## Minggu 3 — Hybrid Model
- [x] Ekstrak feature embedding dari CNN yang udah dilatih
- [x] (Opsional) Ekstrak radiomics features (First-order statistics & Morphological Texture Fusion)
- [x] Training ML classifier (Random Forest/XGBoost) di atas fitur tsb
- [x] Bandingkan performa: CNN-only vs Hybrid
- [x] Tuning hyperparameter (grid search / random search)

## Minggu 4 — Explainability & Evaluasi Final
- [x] Implementasi Grad-CAM
- [x] Evaluasi lengkap: precision, recall, F1, AUC-ROC per kelas
- [x] Cross-validation buat pastiin hasil stabil
- [x] Simpan model final (`.h5`/`.pkl`)

## Minggu 5 — Demo & Dokumentasi
- [x] Bikin demo app (Streamlit/Gradio) — upload citra → hasil + Grad-CAM
- [x] Tulis README.md yang jelas
- [x] Siapin slide presentasi (kalau buat lomba)
- [x] Review ulang PRD vs hasil aktual — ada gap?

## Minggu 6 — Polish & Submit
- [x] Cek ulang semua metrik vs target di PRD
- [x] Bersih-bersih kode & notebook
- [x] Finalisasi dokumentasi (`ARCHITECTURE.md`, `SKILL.md`)
- [x] Submit ke lomba / upload ke GitHub sebagai portofolio

---
**Catatan**: kalau waktu mepet, prioritaskan baseline CNN yang solid dulu (Minggu 1-2) sebelum masuk hybrid — model yang jalan lebih baik dari model ambisius yang ga selesai.
