# Lung Cancer Detection AI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suzirz/lung-cancer-detection-ai/blob/main/notebooks/train_colab_t4.ipynb)

Sistem AI deteksi dini kanker paru dari citra CT scan (3 kelas: *Benign*, *Malignant*, *Normal*) berbasis Transfer Learning CNN & Hybrid Machine Learning.

---

## Cara Menjalankan Training di Google Colab (Tesla T4 GPU)

Klik lencana di atas atau buka tautan langsung:
👉 **[Buka di Google Colab](https://colab.research.google.com/github/suzirz/lung-cancer-detection-ai/blob/main/notebooks/train_colab_t4.ipynb)**

Notebook sudah disiapkan lengkap dengan step-by-step:
1. Pengecekan GPU NVIDIA Tesla T4 (`nvidia-smi`)
2. Clone repository & install requirements
3. Download otomatis dataset (12.184 citra CT scan)
4. Split stratified dataset (Train/Val/Test)
5. Training model baseline CNN dengan **FP16 Mixed Precision**
6. Menampilkan Confusion Matrix & skor evaluasi medis
7. Tombol download model hasil training (`baseline_efficientnet_b0_best.pth`)

---

## Menjalankan Secara Manual di Terminal / VM

```bash
# 1. Clone repository
git clone https://github.com/suzirz/lung-cancer-detection-ai.git
cd lung-cancer-detection-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
python src/preprocessing/download_data.py

# 4. Split data
python -m src.preprocessing.split

# 5. Jalankan training
python src/training/train.py --config configs/default.yaml
```

---

## Struktur Proyek

```text
lung-cancer-detection-ai/
├── configs/                # Konfigurasi terpusat (YAML)
├── data/splits/            # Metadata pembagian train/val/test
├── notebooks/              # Jupyter Notebook (Colab T4)
├── reports/                # Laporan grafik & metrik evaluasi
├── src/
│   ├── preprocessing/      # Dataset loader, augmentasi, split
│   ├── models/             # CNN feature extractor & classifier
│   ├── evaluation/         # EDA & metrik diagnostik
│   └── training/           # Training pipeline (CUDA FP16)
├── tests/                  # Unit test suite
└── requirements.txt
```
