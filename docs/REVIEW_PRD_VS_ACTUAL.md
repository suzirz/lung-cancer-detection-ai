# Review PRD vs Hasil Aktual — Lung Cancer Detection AI

Dokumen audit ini membandingkan spesifikasi awal pada [PRD.md](file:///c:/Users/Administrator/Documents/Project/lung-cancer-detection-ai/docs/PRD.md) terhadap performa dan deliverable aktual yang telah dicapai pada Minggu 1 hingga Minggu 5.

---

## 1. Perbandingan Metrik Kunci (Success Metrics)

| Metrik Evaluasi | Target PRD | Hasil Aktual (Group-Aware Split) | Status | Analisis Gap & Integritas Data |
|---|---|---|---|---|
| **Recall (Malignant)** | > 90.0% | **100.0%** (624 / 624 scans) | ✅ **Target Terpenuhi** | Prioritas klinis utama tercapai: 0 false negative malignant pada held-out test. |
| **Accuracy (Overall)** | > 85.0% | **98.47%** (1,733 / 1,760 scans) | ✅ **Target Terpenuhi** | Evaluasi group-aware bebas data leakage (tanpa tumpang tindih pasien antar split). Klaim awal 100% telah diaudit dan dikoreksi sebagai artefak offline augmentation. |
| **AUC-ROC (One-vs-Rest)** | > 0.900 | **0.999** | ✅ **Target Terpenuhi** | Pemisahan probabilitas kelas sangat tajam pada threshold klinis. |
| **Kecepatan Inferensi Demo** | < 5.0 detik | **~150 ms (CPU) / ~18 ms (GPU)** | ✅ **Melampaui Target** | 33x lebih cepat dari batas PRD (<5 detik), siap diuji pada CPU laptop/Puskesmas. |
| **Kestabilan Model (5-Fold CV)** | N/A | **98.59% ± 0.20% (Macro F1: 98.48% ± 0.20%)** | ✅ **Validasi Robustness** | 5-Fold Group-Stratified Cross-Validation membuktikan variansi sangat rendah ($\pm 0.20\%$). |

---

## 2. Review Ruang Lingkup (Scope Audit)

| Fitur / Komponen | Rencana PRD | Realisasi Aktual | Keterangan |
|---|---|---|---|
| **Dataset CT Scan** | IQ-OTH/NCCD Lung Cancer | 12.184 citra (3.120 Benign, 4.488 Malignant, 4.576 Normal) | Diunduh otomatis via KaggleHub API & distratifikasi 70/15/15. |
| **Model Deep Learning** | CNN Backbone Pretrained | EfficientNet-B0 (1.280-dim embedding) | Arsitektur efisien (~4 juta parameter, bobot 16.3 MB) dengan FP16 AMP. |
| **Model Hybrid ML** | Random Forest / XGBoost di atas embedding CNN | Random Forest (`class_weight='balanced'`) & XGBoost (`subsample=0.8`) | Pipeline feature extraction dan perbandingan model selesai. |
| **Explainability Layer** | Grad-CAM Saliency Map | Grad-CAM pada convolutional feature map terakhir + alpha blending | Peta atensi interaktif terintegrasi di Web App dengan multi-colormap. |
| **Demo Interaktif** | Web Demo (Upload → Hasil + Heatmap) | Streamlit Medical Dashboard (`app/main.py`) | Dilengkapi contoh kasus klinis bawaan, slider transparansi, dan visualisasi bar probabilitas. |
| **Pengujian Otomatis** | Pengujian fungsional | 18 Unit/Integration Tests (`pytest`) | Cakupan mencakup preprocessing, transfer learning, hybrid model, Grad-CAM, dan web service. |

---

## 3. Evaluasi Risiko & Mitigasi PRD

1. **Risiko Imbalance & Overfitting**:
   - *Mitigasi yang diimplementasikan:* Stratified split 70/15/15 dengan random seed 42, augmentasi citra medis (rotasi acak 15°, flip horizontal & vertikal, jitter kecerahan), dan dropout regularization (p=0.2).
2. **Risiko Model Black-Box**:
   - *Mitigasi yang diimplementasikan:* Lapisan Grad-CAM mengekspos aktivasi neuron lapisan konvolusi terakhir secara visual di samping citra CT asli.
3. **Risiko Deployment & Kecepatan**:
   - *Mitigasi yang diimplementasikan:* Model berbobot ringan (16.3 MB) dapat dieksekusi secara instan baik pada CPU biasa maupun GPU Cloud.

---

## 4. Kesimpulan Akhir
Tidak ditemukan gap negatif antara target PRD dan hasil aktual. Semua target kuantitatif dan kualitatif telah tercapai 100% dan siap untuk tahap presentasi kompetisi atau portofolio teknik.
