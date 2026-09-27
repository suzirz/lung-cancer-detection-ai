# Clinical-Grade PulmoScan AI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform PulmoScan AI into an undeniable, hospital-grade Clinical Decision Support System (CDSS) with a conversational AI Co-Pilot, automated PDF clinical reports, uncertainty safeguards, multi-slice volume screening, and external validation protocols.

**Architecture:** A decoupled modular architecture featuring a Streamlit conversational Chatbot interface (ChatGPT/Ollama design), an underlying rule-guided Medical Co-Pilot engine, ReportLab-based EHR PDF generation, confidence temperature calibration, and multi-slice attention pooling.

**Tech Stack:** PyTorch, Torchvision (EfficientNet-B0), Streamlit (Custom Dark Medical Slate), ReportLab (PDF reporting), Scikit-Learn, PyRadiomics / Scikit-Image, Pytest.

**Spec:** Clinical PulmoScan PRD & Medical AI Integrity Standards (Zero-leakage Group CV, Fleischner Society Guidelines).

## Global Constraints

- Zero data leakage across patient cases (`StratifiedGroupKFold`).
- Maximum false negative protection for Malignant cases (Target Recall > 95%).
- Complete explainability: Every diagnosis must pair with Grad-CAM visual heatmaps and radiomics summaries.
- Deterministic test reproducibility: All random seeds fixed (`seed=42`).
- No UI slop: Clean, high-end dark medical slate interface inspired by modern conversational AI (Ollama/ChatGPT).

## Review Focus

1. **Non-CT Image Upload:** If a user uploads an invalid image (e.g. an X-Ray, selfie, or corrupted PNG), the system must reject it with a clear clinical warning instead of guessing.
2. **Borderline Uncertainty:** If class confidence is ambiguous (e.g., 48% vs 52%), the system must trigger a "Borderline / Inconclusive" alert rather than asserting a false diagnosis.
3. **Empty Chat Input:** Handling user conversational prompts gracefully with contextual medical guidance even without an active image.
4. **PDF Render Crash:** Handling missing patient metadata or low-res images cleanly during PDF report compilation without crashing the web app.
5. **Multi-Slice Memory Overflow:** Batching multi-slice volume inference efficiently so memory does not exceed standard CPU/GPU limits.

---

### Task 1: Conversational Clinical AI Co-Pilot UI (`app/chatbot_engine.py` & `app/main.py`)

**Files:**
- Create: `app/chatbot_engine.py`
- Modify: `app/main.py`
- Test: `tests/test_chatbot_engine.py`

**Interfaces:**
- Consumes: `InferenceService.predict_image(image: Image.Image)`
- Produces: `ChatbotEngine.process_message(user_text: str, current_image: Optional[Image.Image], session_state: dict) -> ChatResponse`

- [ ] **Step 1: Write failing test for ChatbotEngine**
```python
# tests/test_chatbot_engine.py
from PIL import Image
from app.chatbot_engine import ChatbotEngine, ChatResponse

def test_chatbot_engine_text_greeting():
    engine = ChatbotEngine()
    response = engine.respond(user_message="Halo, apa fungsi PulmoScan?", image=None)
    assert isinstance(response, ChatResponse)
    assert "skrining CT scan paru" in response.text.lower() or "pulmoscan" in response.text.lower()
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_chatbot_engine.py -v`
Expected: FAIL with ModuleNotFoundError or AttributeError.

- [ ] **Step 3: Implement `ChatbotEngine` with clinical dialogue templates**
Implement `app/chatbot_engine.py` supporting clinical question-answering, Fleischner guideline guidance, Grad-CAM explanation breakdown, and chat history management.

- [ ] **Step 4: Update `app/main.py` with modern ChatGPT/Ollama-style chat interface**
Implement `st.chat_message`, `st.chat_input`, file uploader inside chat bar, sticky sidebar for patient info, and interactive message rendering.

- [ ] **Step 5: Run tests and verify app passes**
Run: `pytest tests/test_chatbot_engine.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**
```bash
git add app/chatbot_engine.py app/main.py tests/test_chatbot_engine.py
git commit -m "feat(ui): implement modern conversational medical co-pilot interface"
```

---

### Task 2: Automated Hospital-Grade PDF Clinical Report Generator (`src/evaluation/report_generator.py`)

**Files:**
- Create: `src/evaluation/report_generator.py`
- Modify: `app/main.py`
- Test: `tests/test_report_generator.py`

**Interfaces:**
- Consumes: `prediction: PredictionResult`, `gradcam_img: Image.Image`, `radiomics_data: dict`
- Produces: `generate_clinical_pdf(patient_info: dict, prediction_data: dict, output_path: str) -> str`

- [ ] **Step 1: Write failing test for PDF generator**
```python
# tests/test_report_generator.py
import os
from PIL import Image
from src.evaluation.report_generator import generate_clinical_pdf

def test_generate_clinical_pdf(tmp_path):
    out_pdf = str(tmp_path / "test_report.pdf")
    dummy_img = Image.new("RGB", (224, 224), color="gray")
    data = {
        "patient_id": "P-9842",
        "predicted_class": "Malignant cases",
        "confidence": 0.985,
        "probabilities": {"Benign cases": 0.01, "Malignant cases": 0.985, "Normal cases": 0.005},
        "radiomics": {"Contrast": 12.4, "Homogeneity": 0.88, "Entropy": 4.12}
    }
    pdf_path = generate_clinical_pdf(patient_data=data, original_img=dummy_img, cam_img=dummy_img, output_path=out_pdf)
    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 1000
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_report_generator.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `generate_clinical_pdf` with ReportLab**
Create professional medical report layout: Hospital Header, Patient Demographics, Original Slice vs Grad-CAM Overlay, Class Distribution Bar, Radiomic Summary, Fleischner Follow-up recommendation, Doctor Signature section, and CE-MDR/FDA disclaimer.

- [ ] **Step 4: Integrate PDF Download button into the Streamlit Chatbot UI**
Provide a 1-click "Download Formal Medical Report (PDF)" button in the chat stream when diagnosis finishes.

- [ ] **Step 5: Run tests and commit**
Run: `pytest tests/test_report_generator.py -v`
```bash
git add src/evaluation/report_generator.py app/main.py tests/test_report_generator.py
git commit -m "feat(report): add automated hospital-grade clinical PDF report generator"
```

---

### Task 3: Confidence Calibration & Borderline Case Safeguard (`src/evaluation/uncertainty.py`)

**Files:**
- Create: `src/evaluation/uncertainty.py`
- Modify: `app/inference_service.py`
- Test: `tests/test_uncertainty.py`

**Interfaces:**
- Consumes: `raw_logits: torch.Tensor` or `probabilities: Dict[str, float]`
- Produces: `UncertaintyResult(is_borderline: bool, status: str, confidence_score: float, recommendation: str)`

- [ ] **Step 1: Write failing test for Uncertainty & OOD validation**
```python
# tests/test_uncertainty.py
from src.evaluation.uncertainty import evaluate_diagnostic_certainty

def test_borderline_detection():
    # Ambiguous probabilities (near 50/50)
    probs = {"Benign cases": 0.49, "Malignant cases": 0.51, "Normal cases": 0.0}
    res = evaluate_diagnostic_certainty(probs, entropy_threshold=0.85)
    assert res.is_borderline is True
    assert "second opinion" in res.recommendation.lower()
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_uncertainty.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `evaluate_diagnostic_certainty`**
Compute Shannon entropy across output probabilities. If entropy is high or top probability margin is below threshold (margin < 0.15), flag as `Borderline / Inconclusive Case`.

- [ ] **Step 4: Integrate safeguard into `InferenceService` and UI**
Present clear alert banner in UI when a case is borderline, ensuring the model never overconfidently misdiagnoses subtle lesions.

- [ ] **Step 5: Run tests and commit**
```bash
git add src/evaluation/uncertainty.py app/inference_service.py tests/test_uncertainty.py
git commit -m "feat(safety): add Shannon entropy uncertainty calibration and borderline safeguards"
```

---

### Task 4: Multi-Slice CT Volume Screening Aggregator (`src/dicom/series_aggregator.py`)

**Files:**
- Create: `src/dicom/series_aggregator.py`
- Modify: `app/main.py`
- Test: `tests/test_series_aggregator.py`

**Interfaces:**
- Consumes: List of `(slice_index: int, image: Image.Image)`
- Produces: `VolumeSummary(max_malignant_slice: int, volume_class: str, slice_predictions: List[dict])`

- [ ] **Step 1: Write failing test for Series Aggregator**
```python
# tests/test_series_aggregator.py
from PIL import Image
from src.dicom.series_aggregator import aggregate_patient_volume

def test_series_aggregator():
    dummy_slices = [(i, Image.new("RGB", (224, 224))) for i in range(10)]
    # Mock predictions
    mock_preds = [{"slice_idx": i, "Malignant cases": 0.1 if i != 5 else 0.95} for i in range(10)]
    summary = aggregate_patient_volume(mock_preds)
    assert summary["suspected_malignant"] is True
    assert summary["peak_slice_index"] == 5
```

- [ ] **Step 2: Run test and verify it fails**
Run: `pytest tests/test_series_aggregator.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `aggregate_patient_volume` and multi-slice batch processor**
Max-pooling over series slices to identify highest-risk regions across a full CT scan volume.

- [ ] **Step 4: Connect multi-file uploader in UI**
Allow uploading multiple slices or a patient scan folder in the chatbot UI.

- [ ] **Step 5: Run tests and commit**
```bash
git add src/dicom/series_aggregator.py tests/test_series_aggregator.py
git commit -m "feat(volume): implement multi-slice 3D CT volume screening aggregator"
```

---

### Task 5: External Benchmark Protocol & Clinical Reader Study Documentation (`docs/clinical_validation.md`)

**Files:**
- Create: `docs/clinical_validation.md`
- Create: `src/evaluation/external_benchmark.py`
- Test: `tests/test_external_benchmark.py`

**Interfaces:**
- Produces: Formal validation framework comparing PulmoScan against NIH LIDC-IDRI standard and defining a multi-reader clinical study protocol.

- [ ] **Step 1: Write external benchmark evaluator script**
Implement reproducible script to download and evaluate public zero-shot LIDC-IDRI benchmark slices.

- [ ] **Step 2: Draft Clinical Reader Study Whitepaper (`docs/clinical_validation.md`)**
Document clinical trial protocol: Radiologist reading time reduction (minutes/scan), Inter-observer agreement (Fleiss' Kappa), Sensitivity increase with AI-assist.

- [ ] **Step 3: Verify and commit**
```bash
git add docs/clinical_validation.md src/evaluation/external_benchmark.py
git commit -m "docs: establish clinical reader study protocol and external validation spec"
```

---
