# ARCHITECTURE — Lung Cancer Detection AI

## 1. Overview Pipeline

```
CT Scan Input (DICOM/PNG)
        |
        v
[1] Preprocessing
    - Resize & normalisasi
    - Windowing (Hounsfield Unit paru)
    - (opsional) Segmentasi paru (U-Net)
        |
        v
[2] Deep Learning Stage (Feature Extractor)
    - CNN (Transfer Learning: EfficientNetB0 / ResNet50)
    - Output: feature embedding vector (bukan langsung klasifikasi final)
        |
        v
[3] Feature Fusion (opsional, advanced)
    - Gabungkan embedding CNN + radiomics features (pyradiomics: shape, texture, intensity)
        |
        v
[4] ML Classifier Stage
    - Random Forest / XGBoost / SVM
    - Input: feature vector dari tahap [2]/[3]
    - Output: kelas (Normal / Benign / Malignant)
        |
        v
[5] Explainability Layer
    - Grad-CAM pada CNN → heatmap area yang jadi fokus model
        |
        v
[6] Output
    - Kelas prediksi + confidence score + heatmap visualisasi
```

## 2. Kenapa Hybrid (DL + ML)?
- CNN unggul di pattern recognition dari citra, tapi cenderung black-box dan butuh data besar
- ML klasik (RF/XGBoost) lebih interpretable, robust di dataset kecil, dan bisa dikombinasikan dengan fitur radiomics yang punya makna klinis (ukuran, bentuk, tekstur nodul)
- Kombinasi ini juga jadi nilai jual di lomba: bukan sekadar "training CNN terus selesai"

## 3. Struktur Folder (repo)

```
lung-cancer-detection-ai/
├── data/
│   ├── raw/                # dataset asli (CT scan)
│   ├── processed/          # hasil preprocessing
│   └── splits/             # train/val/test split
├── notebooks/               # eksplorasi & eksperimen
├── src/
│   ├── preprocessing/       # resize, windowing, segmentasi
│   ├── models/
│   │   ├── cnn_extractor.py # CNN feature extractor
│   │   └── ml_classifier.py # RF/XGBoost classifier
│   ├── explainability/      # Grad-CAM
│   ├── training/            # training loop & config
│   └── evaluation/          # metrics, confusion matrix, ROC
├── app/                     # demo web (Streamlit/Gradio/Flask)
├── docs/                    # dokumentasi (folder ini)
├── models/                  # saved model weights (.h5/.pkl)
├── requirements.txt
└── README.md
```

## 4. Tech Stack (As-Built)
| Layer | Tools & Libraries | Keterangan Implementasi |
|---|---|---|
| Deep Learning | PyTorch 2.x / torchvision | EfficientNet-B0 backbone (4.01M parameter, FP16 AMP) |
| Machine Learning | scikit-learn, XGBoost | Random Forest (`class_weight='balanced'`) & XGBoost (`subsample=0.8`) |
| Image Processing | Pillow, OpenCV | Normalisasi ImageNet, adaptif resize 224x224, colormap Jet/Viridis |
| Explainability | Grad-CAM (Custom PyTorch hook) | Forward/backward hooks pada convolutional feature map terakhir (`features[-1]`) |
| Compute / Hardware | Google Colab / Tesla T4 GPU | 16GB VRAM, CUDA 12.x, PyTorch AMP Mixed Precision |
| Demo App | Streamlit 1.40+ | Dark Medical Slate theme, real-time Grad-CAM overlay, built-in clinical samples |
| Testing Suite | Pytest 9.x | 18 unit/integration tests mencakup preprocessing, hybrid ML, Grad-CAM, & app |

## 5. Alur Data (Train vs Inference As-Built)
- **Training Pipeline**:
  1. Unduh dataset 12.184 citra via KaggleHub API.
  2. Stratified Split 70% Train, 15% Val, 15% Test dengan seed 42.
  3. Preprocessing & Augmentasi medis (rotasi 15°, flip H/V, normalisasi).
  4. Training EfficientNet-B0 dengan FP16 AMP, checkpoint best validation recall.
  5. Ekstraksi 1.280-dim vektor embedding dari `model.extract_features()`.
  6. Pelatihan ensemble Random Forest & XGBoost di atas embedding.
  7. Evaluasi komparatif multi-metrik, ROC-AUC multiclass, dan 5-Fold Cross-Validation.
- **Inference Pipeline (Streamlit Web Demo)**:
  1. Input: Citra CT scan aksial (dari uploader file atau sampel klinis bawaan).
  2. Preprocessing evaluasi: konversi RGB, resize 224x224, normalisasi ImageNet mean/std.
  3. Forward pass pada model CNN untuk logits klasifikasi dan probabilitas softmax.
  4. Backward pass Grad-CAM pada target class logit untuk menghasilkan 2D heatmap.
  5. Blending heatmap dengan colormap terpilih ke atas citra asli menggunakan parameter alpha yang dapat diatur pengguna.
  6. Output: Diagnosis klinis (Normal/Benign/Malignant), keyakinan (%), latensi inferensi (~150 ms), visualisasi Grad-CAM, dan disclaimer medis.

