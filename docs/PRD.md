# PRD — Lung Cancer Detection AI

## 1. Latar Belakang
Deteksi dini kanker paru dari CT scan masih sangat bergantung pada interpretasi manual radiolog. Ini rawan human error, butuh waktu lama, dan akses ke radiolog spesialis terbatas terutama di daerah. Project ini membangun sistem AI (hybrid Deep Learning + Machine Learning) untuk membantu skrining awal nodul paru dari citra CT scan.

## 2. Tujuan
- Membangun model yang bisa mengklasifikasikan citra CT scan paru ke kategori **normal / benign / malignant**
- Menghasilkan sistem yang *explainable* (bukan cuma black-box) — cocok buat dipresentasikan di kompetisi
- Deliverable akhir: model + demo interaktif + dokumentasi teknis

## 3. Target Pengguna
- Juri kompetisi (SIC8 atau sejenisnya) — audiens teknis + non-teknis
- Radiolog / tenaga medis sebagai *decision support*, bukan pengganti diagnosis
- Diri sendiri — sebagai portofolio AI/ML

## 4. Scope
### In-scope
- Preprocessing CT scan (DICOM/PNG dataset)
- Model klasifikasi hybrid: CNN (feature extraction) + ML klasik (RF/XGBoost) untuk klasifikasi akhir
- Evaluasi model dengan metrik medis (recall/sensitivity jadi prioritas)
- Explainability (Grad-CAM)
- Demo web sederhana (upload citra → hasil prediksi + visualisasi area fokus model)

### Out-of-scope
- Deployment ke pasien/klinik nyata
- Sertifikasi medis/regulasi (FDA/BPOM dsb)
- Deteksi real-time dari alat CT scan langsung

## 5. Dataset
- **Utama**: IQ-OTH/NCCD Lung Cancer Dataset (Kaggle) — sudah terklasifikasi normal/benign/malignant
- **Opsional (advanced)**: LIDC-IDRI — anotasi nodul dari 4 radiolog, lebih kompleks

## 6. Success Metrics
| Metrik | Target |
|---|---|
| Recall (sensitivity) | > 90% — prioritas utama, false negative fatal di kasus medis |
| Accuracy | > 85% |
| AUC-ROC | > 0.90 |
| Demo | Bisa upload citra & dapat hasil < 5 detik |

## 7. Risiko & Mitigasi
- **Dataset kecil/imbalance** → data augmentation, class weighting
- **Overfitting** → transfer learning, dropout, cross-validation
- **Model dianggap "black box"** → wajib ada Grad-CAM/explainability layer
- **Waktu mepet (lomba)** → mulai dari baseline CNN dulu sebelum hybrid

## 8. Timeline Singkat
Lihat `TODO.md` untuk breakdown task per minggu.
