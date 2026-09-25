# Contributing to PulmoScan

Thank you for your interest in contributing to PulmoScan! This project is an open-source research and engineering framework for lung nodule classification and explainability on CT scans. We welcome contributions from machine learning engineers, clinical researchers, and radiologists.

---

## Areas Where Contributions Are Needed

1. **External Cross-Institutional Validation**:
   - Evaluating pretrained weights on external cohorts (e.g., LIDC-IDRI, LUNA16, NLST).
   - Validating Hounsfield Unit (HU) windowing and 3D volumetric pipelines.
2. **Clinical Saliency Audits**:
   - Radiologist reviews of Grad-CAM heatmaps against true pathological nodule boundaries.
   - Identifying false-positive drivers (e.g., post-tuberculosis fibrosis, calcified granulomas, pleural thickening).
3. **Engineering & Edge Deployment**:
   - ONNX Runtime and TensorRT optimization for edge hospital inference nodes.
   - Native DICOM C-STORE / C-MOVE PACS receivers using `pydicom` / `pynetdicom`.

---

## Development Workflow

### 1. Clone & Setup Environment
```bash
git clone https://github.com/suzirz/lung-cancer-detection-ai.git
cd lung-cancer-detection-ai
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite
Before opening a pull request, verify that all automated unit and integration tests pass:
```bash
python -m pytest tests/ -v
```

### 3. Data Leakage Invariant
Any modification to data loading or splitting **must** preserve group-aware patient isolation. Augmented variants of the same patient CT scan must never cross split boundaries. The test suite automatically asserts:
$$\text{Groups}(\text{Train}) \cap \text{Groups}(\text{Val}) = \emptyset$$
$$\text{Groups}(\text{Train}) \cap \text{Groups}(\text{Test}) = \emptyset$$

---

## Submitting Pull Requests

1. Create a feature branch: `git checkout -b feature/your-feature-name`.
2. Commit changes with clear, descriptive messages.
3. Ensure CI passes on all matrix configurations (Python 3.10, 3.11).
4. Submit a Pull Request targeting `main`.
