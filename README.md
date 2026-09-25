# PulmoScan: Lung Cancer CT Detection & Explainability Framework

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suzirz/lung-cancer-detection-ai/blob/main/notebooks/train_colab_t4.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/Tests-23%20Passing-brightgreen.svg)

> [!WARNING]
> **RESEARCH & EDUCATIONAL PROTOTYPE · NOT FOR CLINICAL USE**
> PulmoScan is an academic feasibility prototype and machine learning software engineering demonstration. It is **not** a cleared medical device and is **not intended for clinical diagnosis, patient management, or surgical guidance**. All models were evaluated on public research datasets and require independent multi-center clinical validation before any diagnostic use.

PulmoScan provides an end-to-end computer vision and explainable AI pipeline for classifying pulmonary nodules on chest computed tomography (CT) scans into three diagnostic categories: **Normal parenchyma**, **Benign nodule**, and **Malignant neoplasm**. The system combines transfer-learned convolutional neural networks (EfficientNet-B0), quantitative intensity/gradient radiomics, tree-based tabular classifiers (Random Forest & XGBoost), and gradient-weighted class activation mapping (Grad-CAM).

---

## Technical Status: Trained Weights vs. Pipeline Components

| Component | Category | Current Status | Description |
|---|---|---|---|
| **EfficientNet-B0 Backbone** | Deep Learning | Trained Checkpoint (`baseline_efficientnet_b0_best.pth`) | 4.01M parameter CNN extracting 1,280-dim feature vectors and 3-class logits |
| **Hybrid Random Forest** | Machine Learning | Trained Model (`models/hybrid_random_forest.joblib`) | Balanced class-weight tree ensemble operating on 1,280-dim embeddings |
| **Hybrid XGBoost** | Machine Learning | Trained Model (`models/hybrid_xgboost.joblib`) | Gradient-boosted decision trees (`subsample=0.8`) with softmax multi-class objective |
| **Radiomics Extractor** | Feature Engineering | Active (`src/models/radiomics_extractor.py`) | 13 first-order intensity moments, Shannon entropy, and morphological gradient descriptors |
| **Grad-CAM Visualizer** | Explainability | Active (`src/explainability/gradcam.py`) | Backward-hook saliency mapping targeting final convolution stage (`model.features[-1]`) |
| **Interactive Triage Web App** | Diagnostic Demo | Active (`app/main.py`) | Streamlit interface with dual image comparison, alpha slider, and sample selector |
| **End-to-End Colab Pipeline** | Cloud Acceleration | Active (`notebooks/train_colab_t4.ipynb`) | 1-click reproducible pipeline on NVIDIA Tesla T4 with FP16 Automatic Mixed Precision |

---

## Empirical Benchmark (1,828 Held-Out Test Scans)

Empirical evaluation results on the held-out test cohort (1,828 independent CT scans, strictly isolated before training):

| Model Architecture | Task | Test Samples | Accuracy | Macro Precision | Macro Recall | Macro F1 | Latency (CPU) |
|---|---|---|---|---|---|---|---|
| **EfficientNet-B0 (CNN)** | 3-Class End-to-End | 1,828 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | ~150 ms |
| **Hybrid: CNN + XGBoost** | Embedding Classifier | 1,828 | **99.94%** | 99.95% | 99.93% | 99.94% | ~165 ms |
| **Hybrid: CNN + Random Forest** | Embedding Classifier | 1,828 | **99.89%** | 99.88% | 99.89% | 99.89% | ~160 ms |

### Confusion Matrix Breakdown (EfficientNet-B0, 1,828 Held-Out Test Cases):
- **Benign Cases (468 scans)**: 468 True Positives, 0 misclassified as Malignant, 0 as Normal.
- **Malignant Cases (673 scans)**: 673 True Positives, 0 misclassified as Benign, 0 as Normal.
- **Normal Parenchyma (687 scans)**: 687 True Negatives, 0 misclassified as Benign, 0 as Malignant.

> **Clinical Reality Check**: Zero false negatives were observed on this augmented benchmark test set. While these metrics confirm model convergence and high separability on the IQ-OTHNCCD dataset, real clinical deployment faces domain shifts across slice thicknesses (1 mm thin-slice vs 5 mm thick-slice), reconstruction kernels (bone vs lung window), and scanner manufacturers (Siemens, GE, Philips). Cross-institutional validation on datasets like LIDC-IDRI is required before clinical use.

![Confusion Matrix](reports/baseline_confusion_matrix.png)

![Multi-Class One-vs-Rest ROC Curves](reports/roc_auc_curve.png)

![Model Comparison Benchmark](reports/model_comparison.png)

---

## Explainable AI: Grad-CAM Clinical Saliency Maps

In clinical decision support, opaque predictions without spatial grounding cannot be verified by radiologists. PulmoScan implements Gradient-weighted Class Activation Mapping (Grad-CAM) attached to the final convolutional feature extractor (`model.features[-1]`).

The gradient of the target class score $y^c$ with respect to feature activation map $A^k$ is computed:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i, j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

![Grad-CAM Saliency Map Showcase](reports/gradcam_showcase.png)

### Observed Attention Patterns:
1. **Malignant Cases**: Model gradients concentrate on dense irregular masses, spiculated margins, and pleural retractions.
2. **Benign Cases**: Attention centers on well-circumscribed, round/oval parenchymal opacities without invasive margin spread.
3. **Normal Cases**: Gradients remain diffuse across normal vascular trees without localized hotspots.

---

## Reference Dataset (12,184 CT Scans, IQ-OTHNCCD)

The system is trained and evaluated on the augmented IQ-OTHNCCD lung cancer CT dataset:

| Diagnostic Class | Training Set (70%) | Validation Set (15%) | Testing Set (15%) | Total Scans |
|---|---|---|---|---|
| **Benign Cases** | 2,184 | 468 | 468 | **3,120 scans** |
| **Malignant Cases** | 3,142 | 673 | 673 | **4,488 scans** |
| **Normal Parenchyma** | 3,202 | 687 | 687 | **4,576 scans** |
| **Total Cohort** | **8,528 scans** | **1,828 scans** | **1,828 scans** | **12,184 scans** |

### Data Pipeline & Preprocessing Protocol:
1. **Resolution & Normalization**: Slices are rescaled to 224x224 pixels and normalized to ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
2. **Data Leakage Prevention**: Stratified splitting (`data/splits/*.json`) was performed with fixed random seed (42) prior to any training or feature extraction.
3. **Clinical Domain Augmentation**:
   - Random horizontal flips ($p = 0.5$) and vertical flips ($p = 0.3$).
   - Random rotations ($\pm 15^\circ$) to simulate patient positioning variation in scanner gantries.
   - Micro color jitter (brightness 0.1, contrast 0.1) simulating tube current and mAs differences.

---

## Multimodal Architecture & Radiomics Fusion

```text
Axial CT Slice (224x224)
        │
        ├─────────────────────────────────────────────────┐
        ▼                                                 ▼
[1] CNN Feature Extractor (EfficientNet-B0)     [2] Quantitative Radiomics
    ├── 16 Inverted Residual Blocks (MBConv)        ├── First-Order Moments (Mean, Std, Skew, Kurt)
    ├── Adaptive Global Average Pooling             ├── Shannon Entropy & Uniform Energy
    └── 1,280-Dimensional Latent Embedding          └── Morphological Gradients (Laplacian Var, Sobel)
        │                                                 │
        ├──────────────────────┬──────────────────────────┘
        ▼                      ▼
[3] End-to-End Logits     [4] Multimodal Feature Fusion
    ├── Dropout (p=0.2)        └── Concatenated Feature Vector (1,293 dimensions)
    └── Linear Layer (3 cls)          ├── Random Forest Classifier (balanced weights)
        │                             └── XGBoost Gradient Boosted Trees
        ▼
[5] Explainability Engine (Grad-CAM)
    └── Saliency Map Overlaid on CT Slice (Jet / Viridis colormaps)
```

### Radiomics Descriptors (`src/models/radiomics_extractor.py`):
- **Intensity Histogram**: Mean, variance, standard deviation, skewness, kurtosis, and energy ($E = \frac{1}{N}\sum x_i^2$).
- **Tissue Complexity**: Shannon entropy ($H = -\sum p_i \log_2 p_i$) on 64-bin intensity distributions.
- **Nodule Boundary Sharpness**: Laplacian variance ($\sigma_{\text{Laplace}}^2$) quantifying edge abruptness versus ground-glass blurring.
- **Gradient Distribution**: Sobel filter spatial derivatives ($\mu_{\text{Sobel}}, \sigma_{\text{Sobel}}$) and interquartile range (IQR).

---

## Interactive Diagnostic Web Application

PulmoScan includes an interactive medical dashboard built with Streamlit (`app/main.py`):

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch interactive application
streamlit run app/main.py
```

Open `http://localhost:8501` in your browser.

### Key Capabilities:
- **Clinical Sample Library**: Immediate testing with pre-loaded, verified Benign, Malignant, and Normal CT slices.
- **Custom Image Upload**: Upload any PNG or JPG axial CT scan for classification.
- **Side-by-Side Saliency Verification**: View original scans alongside Grad-CAM overlays with variable opacity ($\alpha = 0.10 \text{ to } 0.90$) and colormaps (*Jet, Inferno, Viridis, Magma*).
- **Latency Measurement**: Real-time inference benchmark displays processing speed (~150 ms on CPU).

---

## 1-Click Cloud Training (Google Colab Tesla T4)

To retrain the complete pipeline on an NVIDIA Tesla T4 GPU:

👉 **[Open In Google Colab](https://colab.research.google.com/github/suzirz/lung-cancer-detection-ai/blob/main/notebooks/train_colab_t4.ipynb)**

The notebook automatically executes:
1. GPU hardware check (`nvidia-smi`).
2. Repository clone and environment setup.
3. Dataset acquisition via KaggleHub API (12,184 images).
4. EfficientNet-B0 training with PyTorch AMP FP16 (`torch.cuda.amp`).
5. Feature extraction and Random Forest / XGBoost training.
6. Generation of ROC curves, confusion matrices, and Grad-CAM heatmaps.
7. Model artifact download (`.pth` and `.joblib`).

---

## Repository Structure

```text
lung-cancer-detection-ai/
├── app/
│   ├── main.py                  # Streamlit diagnostic web application
│   ├── inference_service.py     # Thread-safe inference & Grad-CAM service
│   └── samples/                 # Sample clinical CT slices (Benign, Malignant, Normal)
├── configs/
│   └── default.yaml             # Hyperparameters, batch size, seed, and data paths
├── data/
│   └── splits/                  # Train/Val/Test stratified splits (JSON)
├── models/
│   └── baseline_efficientnet_b0_best.pth  # Trained model weights (16.3 MB)
├── notebooks/
│   └── train_colab_t4.ipynb     # Reproducible Google Colab T4 training notebook
├── reports/
│   ├── baseline_confusion_matrix.png     # Test set confusion matrix
│   ├── roc_auc_curve.png                 # Multi-class ROC curves
│   ├── model_comparison.png              # Architecture performance comparison
│   ├── gradcam_showcase.png              # 3-case Grad-CAM visual demonstration
│   └── eda/                              # Class balance & raw sample plots
├── scripts/
│   └── generate_readme_assets.py         # Visual asset generation script
├── src/
│   ├── preprocessing/           # Dataset loader, augmentations, and stratified split
│   ├── models/                  # CNN extractor, hybrid classifiers, radiomics module
│   ├── explainability/          # Grad-CAM implementation & heatmap blending
│   ├── evaluation/              # Clinical recall, ROC-AUC, 5-fold cross-validation
│   └── training/                # Training loop with PyTorch AMP FP16
├── tests/                       # Pytest test suite (23 unit & integration tests)
└── requirements.txt             # Project dependencies
```

---

## Verification & Testing Suite

All modules are covered by 23 automated unit and integration tests:

```bash
python -m pytest tests/ -v
```

### Test Coverage Summary:
- **Preprocessing (`test_preprocessing.py`)**: Split ratios, stratifications, and DataLoader batches.
- **CNN Architecture & Metrics (`test_model_and_metrics.py`)**: Forward pass, feature dimension (1,280), medical recall calculation, confusion matrix.
- **Hybrid Modeling (`test_hybrid.py`)**: Random Forest training, XGBoost training, evaluation export.
- **Explainability & Audits (`test_explainability_and_final_eval.py`)**: Grad-CAM heatmap bounds $[0, 1]$, alpha blending, multi-class ROC-AUC, 5-fold cross-validation.
- **App Inference Service (`test_app_inference.py`)**: Checkpoint loading, probability normalization ($\sum p_i = 1.0$), real sample evaluation, error handling.
- **Radiomics & Fusion (`test_radiomics.py`)**: First-order intensity metrics, uniform image variance handling, multimodal concatenation.

---

## License

This project is distributed under the [MIT License](LICENSE).
