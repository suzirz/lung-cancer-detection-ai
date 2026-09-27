"""
Unit tests for Multi-Slice 3D CT Volume Screening Aggregator.
File: tests/test_series_aggregator.py
"""

import pytest
from PIL import Image

from src.dicom.series_aggregator import aggregate_patient_volume, VolumeSummary


def test_aggregate_patient_volume_malignant_detected():
    # Simulasi 10 slice CT Scan toraks pasien
    mock_slice_preds = [
        {"slice_idx": i, "predicted_class": "Normal", "confidence": 0.95, "probabilities": {"Normal": 0.95, "Benign": 0.03, "Malignant": 0.02}}
        for i in range(10)
    ]
    # Lesi ganas tampak jelas pada slice 5 dan 6
    mock_slice_preds[5] = {
        "slice_idx": 5,
        "predicted_class": "Malignant",
        "confidence": 0.94,
        "probabilities": {"Normal": 0.01, "Benign": 0.05, "Malignant": 0.94}
    }
    mock_slice_preds[6] = {
        "slice_idx": 6,
        "predicted_class": "Malignant",
        "confidence": 0.98,
        "probabilities": {"Normal": 0.005, "Benign": 0.015, "Malignant": 0.98}
    }

    summary = aggregate_patient_volume(mock_slice_preds)
    assert isinstance(summary, VolumeSummary)
    assert summary.suspected_malignant is True
    assert summary.peak_slice_index == 6
    assert summary.peak_malignancy_prob == 0.98
    assert summary.total_slices == 10
    assert 5 in summary.high_risk_slices and 6 in summary.high_risk_slices
    assert "slice #6" in summary.summary_narrative.lower()


def test_aggregate_patient_volume_all_normal():
    mock_slice_preds = [
        {"slice_idx": i, "predicted_class": "Normal", "confidence": 0.98, "probabilities": {"Normal": 0.98, "Benign": 0.01, "Malignant": 0.01}}
        for i in range(8)
    ]
    summary = aggregate_patient_volume(mock_slice_preds)
    assert isinstance(summary, VolumeSummary)
    assert summary.suspected_malignant is False
    assert len(summary.high_risk_slices) == 0
    assert summary.overall_diagnosis == "Normal"
