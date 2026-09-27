"""
Unit tests for Clinical PDF Report Generator.
File: tests/test_report_generator.py
"""

import os
from PIL import Image
import pytest

from src.evaluation.report_generator import generate_clinical_pdf


def test_generate_clinical_pdf(tmp_path):
    out_pdf = str(tmp_path / "test_medical_report.pdf")
    orig_img = Image.new("RGB", (224, 224), color=(80, 80, 80))
    cam_img = Image.new("RGB", (224, 224), color=(180, 40, 40))

    patient_data = {
        "patient_id": "TCGA-LU-4819",
        "study_date": "2026-09-27",
        "predicted_class": "Malignant",
        "confidence": 0.985,
        "probabilities": {
            "Benign": 0.010,
            "Malignant": 0.985,
            "Normal": 0.005
        },
        "radiomics": {
            "GLCM Contrast": 14.82,
            "GLCM Homogeneity": 0.841,
            "GLCM Energy": 0.092,
            "GLCM Correlation": 0.612
        },
        "dicom_info": {
            "slice_thickness_mm": 1.25,
            "kvp": 120.0,
            "hu_mean": -650.0
        }
    }

    pdf_path = generate_clinical_pdf(
        patient_data=patient_data,
        original_img=orig_img,
        cam_img=cam_img,
        output_path=out_pdf
    )

    assert os.path.exists(pdf_path)
    assert os.path.getsize(pdf_path) > 2000  # ReportLab PDF with images has non-trivial size
