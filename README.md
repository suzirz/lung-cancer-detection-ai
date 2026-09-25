# PulmoScan: Lung Cancer CT Detection & Explainability Framework

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/suzirz/lung-cancer-detection-ai/blob/main/notebooks/train_colab_t4.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/Tests-29%20Passing-brightgreen.svg)

> [!WARNING]
> **RESEARCH & EDUCATIONAL PROTOTYPE · NOT FOR CLINICAL USE**
> PulmoScan is an academic feasibility prototype and machine learning software engineering demonstration. It is **not** a cleared medical device and is **not intended for clinical diagnosis, patient management, or surgical guidance**. All models were evaluated on public research datasets and require independent multi-center clinical validation before any diagnostic use.

PulmoScan provides an end-to-end computer vision and explainable AI pipeline for classifying pulmonary nodules on chest computed tomography (CT) scans into three diagnostic categories: **Normal parenchyma**, **Benign nodule**, and **Malignant neoplasm**. The system combines transfer-learned convolutional neural networks (EfficientNet-B0), quantitative intensity/gradient radiomics, tree-based tabular classifiers (Random Forest & XGBoost), and gradient-weighted class activation mapping (Grad-CAM).

---

## Problem Context: Why This Matters

Lung cancer kills more Indonesians than any other cancer. The numbers are specific and severe:

- **38,000+ new cases per year** — more than 100 new diagnoses every day across the archipelago.
- **70–90% of patients are diagnosed at Stage III or IV**, when the 5-year survival rate drops below 10%. Early-stage lung cancer (Stage I) has a 5-year survival rate above 70%, but most patients never get caught early.
- **Indonesia has ~1.2 radiologists per 100,000 people**. Specialists and CT scanners are concentrated in Jakarta, Surabaya, and a handful of major cities. Patients in Kalimantan, Sulawesi, Papua, and rural Java often travel hundreds of kilometers for a scan — if they get one at all.
- **No national screening program exists.** Low-dose CT screening (the global gold standard for high-risk smokers) is not covered by JKN (national health insurance) and is only available at select private hospitals.
- **Diagnostic confusion with TB.** Chronic cough and chest pain — the most common early lung cancer symptoms — are routinely attributed to tuberculosis or COPD in primary care, delaying referral by months.

The gap is not just a shortage of scanners. It is a shortage of trained eyes to read the scans. A computer-aided triage tool that flags suspicious nodules and shows the radiologist *where* and *why* it flagged them does not replace the doctor — it extends their reach to the district hospitals and Puskesmas that currently have no specialist coverage.

PulmoScan is a prototype exploring whether a lightweight model (4M parameters, runs on a laptop CPU in ~150ms) can perform this triage function with sufficient accuracy to be worth validating further.

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

## Empirical Benchmark (Group-Aware Splits, Held-Out Test Set)

> **⚠️ METRICS PENDING RE-TRAINING**
> The results below will be updated after re-training with group-aware splits. Previous metrics (100% across all classes) were invalid due to data leakage — see [Methodological Note](#methodological-note-data-leakage-correction) below.

| Model Architecture | Task | Test Samples | Accuracy | Macro Precision | Macro Recall | Macro F1 | Latency (CPU) |
|---|---|---|---|---|---|---|---|
| **EfficientNet-B0 (CNN)** | 3-Class End-to-End | TBD | TBD | TBD | TBD | TBD | ~150 ms |
| **Hybrid: CNN + XGBoost** | Embedding Classifier | TBD | TBD | TBD | TBD | TBD | ~165 ms |
| **Hybrid: CNN + Random Forest** | Embedding Classifier | TBD | TBD | TBD | TBD | TBD | ~160 ms |

![Confusion Matrix](reports/baseline_confusion_matrix.png)

![Multi-Class One-vs-Rest ROC Curves](reports/roc_auc_curve.png)

![Model Comparison Benchmark](reports/model_comparison.png)

---

## Methodological Note: Data Leakage Correction

The IQ-OTHNCCD Augmented Dataset contains 12,184 images generated from approximately 1,000 original patient CT scans through offline augmentation (rotation, flipping, brightness adjustment). Each original scan has roughly 10 augmented variants named with the pattern `{Class} case ({patient_id})({augmentation_id}).jpg`.

**Problem identified**: The initial pipeline split data per-image using `sklearn.train_test_split`, treating each augmented variant as an independent sample. This scattered near-identical images from the same patient across train, validation, and test sets. Quantified overlap:
- 830 patient groups appeared in both training and test splits
- 843 patient groups appeared in both training and validation splits

This leakage allowed the model to "recognize" augmented twins during evaluation, producing artificially perfect metrics.

**Fix applied** ([`split.py`](src/preprocessing/split.py)): Splitting now operates at the **patient/scan group level** using `extract_group_id()` to parse filenames and group all augmented variants of the same scan together. All variants of one scan are assigned to exactly one split partition (train, val, OR test — never across splits). Zero cross-split group overlap is verified by assertion at split time and by automated tests (`tests/test_preprocessing.py`).

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

| Diagnostic Class | Total Scans | Source |
|---|---|---|
| **Benign Cases** | **3,120** | IQ-OTH/NCCD (augmented) |
| **Malignant Cases** | **4,488** | IQ-OTH/NCCD (augmented) |
| **Normal Parenchyma** | **4,576** | IQ-OTH/NCCD (augmented) |
| **Total Cohort** | **12,184** | ~1,000 original scans × ~10 augmentations |

Splits are performed at the **patient/scan group level** (70/15/15 ratio applied to groups, not individual images). Exact image counts per split depend on the number of augmentation variants per patient.

### Data Pipeline & Preprocessing Protocol:
1. **Resolution & Normalization**: Slices are rescaled to 224x224 pixels and normalized to ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
2. **Data Leakage Prevention**: Group-aware stratified splitting (`src/preprocessing/split.py`) ensures all augmented variants of a single patient scan remain in the same partition. Zero cross-split patient overlap is verified by automated test and runtime assertion.
3. **Clinical Domain Augmentation** (applied at training time only):
   - Random horizontal flips ($p = 0.5$) and vertical flips ($p = 0.3$).
   - Random rotations ($\pm 15^\circ$) to simulate patient positioning variation in scanner gantries.
   - Micro color jitter (brightness 0.1, contrast 0.1) simulating tube current and mAs differences.

---

## Design Rationale: Why Hybrid CNN + Tree Classifiers

A reasonable question: why not just train a CNN end-to-end and call it done?

Three concrete reasons drove the hybrid architecture:

**1. Clinical recall matters more than accuracy.** In lung cancer screening, a missed malignant nodule (false negative) has catastrophic consequences — the patient walks away thinking they're healthy. A CNN's softmax layer optimizes for overall accuracy, which can sacrifice recall on the minority class. Random Forest and XGBoost classifiers support native `class_weight="balanced"` and `scale_pos_weight`, letting us explicitly push the decision boundary toward higher malignant recall even at the cost of some false positives. Published benchmarks on similar CT datasets report 3–5% recall improvement on malignant cases when switching from CNN-softmax to CNN-embedding + tree classifier.

**2. Radiomics captures what CNNs learn to ignore.** CNNs excel at learning hierarchical visual features, but they tend to compress low-level statistics (mean intensity, entropy, edge sharpness) into abstract representations that are hard to audit. Radiomics features — first-order intensity moments, Shannon entropy, Laplacian variance, Sobel gradient statistics — are interpretable, clinically meaningful, and provide a complementary signal channel. Concatenating 1,280-dim CNN embeddings with 13 radiomics features gives the tree classifiers information from both abstract pattern recognition and quantitative tissue characterization.

**3. Explainability is not optional in medical AI.** Indonesian radiologists and competition judges cannot accept a model that says "malignant" without showing its reasoning. Grad-CAM provides spatial attribution ("the model focused on this region"), while radiomics provides quantitative attribution ("the Laplacian variance of this region is 2.3x higher than normal tissue"). Together, they give a doctor two independent verification channels — one visual, one numerical.

The hybrid approach is not novel. It follows established methodology from published literature on CT-based lung nodule classification. The contribution here is the integrated pipeline: one command trains the CNN, extracts embeddings, trains the tree classifiers, generates Grad-CAM visualizations, and produces a deployable web demo.

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

All modules are covered by 29 automated unit and integration tests:

```bash
python -m pytest tests/ -v
```

### Test Coverage Summary:
- **Preprocessing (`test_preprocessing.py`)**: Group ID extraction, group-aware split ratios, zero cross-split patient overlap verification, augmentation integrity (all variants stay together), real split leakage detection, and DataLoader batches.
- **CNN Architecture & Metrics (`test_model_and_metrics.py`)**: Forward pass, feature dimension (1,280), medical recall calculation, confusion matrix.
- **Hybrid Modeling (`test_hybrid.py`)**: Random Forest training, XGBoost training, evaluation export.
- **Explainability & Audits (`test_explainability_and_final_eval.py`)**: Grad-CAM heatmap bounds $[0, 1]$, alpha blending, multi-class ROC-AUC, 5-fold cross-validation.
- **App Inference Service (`test_app_inference.py`)**: Checkpoint loading, probability normalization ($\sum p_i = 1.0$), real sample evaluation, error handling.
- **Radiomics & Fusion (`test_radiomics.py`)**: First-order intensity metrics, uniform image variance handling, multimodal concatenation.

---

## Reproducibility

All training runs are deterministic at the framework level:

| Factor | Setting |
|---|---|
| **Random seed** | 42 (fixed across NumPy, PyTorch, sklearn) |
| **Data split** | Group-aware stratified, persisted as JSON (`data/splits/`) |
| **Model weights** | Deterministic initialization from `torchvision` pretrained ImageNet checkpoint |
| **Hardware target** | NVIDIA Tesla T4 (16 GB VRAM), Google Colab |
| **Precision** | FP16 Mixed Precision via `torch.cuda.amp` |

To reproduce from scratch: open the [Colab notebook](https://colab.research.google.com/github/suzirz/lung-cancer-detection-ai/blob/main/notebooks/train_colab_t4.ipynb), select T4 GPU runtime, and run all cells. The pipeline downloads data, splits, trains, and evaluates without manual intervention.

**Caveat on exact reproducibility**: PyTorch does not guarantee bitwise-identical results across GPU architectures or CUDA versions, even with fixed seeds. Metrics may vary by ±0.5% on different hardware. The split files (`data/splits/*.json`) are the canonical reference — as long as the same splits are used, model comparisons remain valid.

---

## Known Limitations

This section documents what PulmoScan **cannot** do and where it will fail. These are not future roadmap items — they are structural constraints of the current design and dataset.

### 1. Single-Source Dataset

All training and evaluation data comes from one institution (Iraq-Oncology Teaching Hospital / National Center for Cancer Diseases). The model has never seen CT scans from any other hospital, scanner, or patient population. Performance on scans from Siemens, GE, Philips, or Toshiba scanners at other institutions is unknown and likely degraded.

### 2. No Scanner or Reconstruction Kernel Normalization

CT scan appearance varies significantly based on:
- **Slice thickness**: The dataset uses a single slice thickness. Models trained on 1mm thin-slice scans perform poorly on 5mm thick-slice scans (and vice versa), because nodule visibility and texture change dramatically.
- **Reconstruction kernel**: Lung kernel (sharp, high-frequency) vs. soft tissue kernel (smooth, low-frequency) produce visually different images of the same anatomy. PulmoScan does not detect or normalize for kernel type.
- **Scanner manufacturer and model**: Each vendor's detector geometry, tube voltage defaults, and post-processing pipelines produce subtly different image characteristics.

The preprocessing pipeline applies only resize + ImageNet normalization. It does not perform HU windowing, kernel harmonization, or slice thickness resampling.

### 3. 2D Slices, Not 3D Volumes

PulmoScan classifies individual 2D axial slices, not 3D CT volumes. In clinical radiology, a nodule's 3D morphology (volume doubling time, spiculation pattern across slices, relationship to bronchi/vessels) carries critical diagnostic information that single-slice analysis misses entirely.

### 4. Augmentation-Inflated Dataset

The 12,184 images originate from approximately 1,000 unique scans, each augmented ~10x offline. This means:
- The effective training diversity is ~1,000 patients, not 12,184.
- The model's exposure to anatomical variation is limited to whatever variation existed in the original ~1,000 scans.
- The group-aware split prevents leakage, but does not increase real sample diversity.

### 5. No DICOM Metadata

The dataset provides JPEG images stripped of DICOM headers. PulmoScan has no access to patient age, sex, scan parameters, slice position, or clinical history — all of which inform real diagnostic decisions.

### 6. Three-Class Simplification

Real pulmonary nodule classification involves a spectrum: pure ground-glass opacity (GGO), part-solid, solid, calcified, cavitary, with sub-classifications by size (< 6mm, 6–8mm, > 8mm) following Fleischner Society guidelines. PulmoScan reduces this to three coarse categories (Benign / Malignant / Normal), which does not reflect clinical practice.

### 7. No Multi-Center Validation

The model has not been evaluated on external datasets (LIDC-IDRI, LUNA16, NLST). Until cross-institutional validation is performed, reported metrics apply only to IQ-OTHNCCD data and should not be extrapolated to other populations or imaging protocols.

---

## License

This project is distributed under the [MIT License](LICENSE).
