"""
Unit tests for Diagnostic Uncertainty & Borderline Safeguards.
File: tests/test_uncertainty.py
"""

import pytest
import numpy as np
from PIL import Image

from src.evaluation.uncertainty import (
    evaluate_diagnostic_certainty,
    UncertaintyResult,
    validate_image_quality
)


def test_confident_prediction():
    probs = {"Benign": 0.02, "Malignant": 0.97, "Normal": 0.01}
    res = evaluate_diagnostic_certainty(probs)
    assert isinstance(res, UncertaintyResult)
    assert res.is_borderline is False
    assert res.certainty_level == "High"
    assert res.entropy < 0.3


def test_borderline_prediction():
    probs = {"Benign": 0.48, "Malignant": 0.51, "Normal": 0.01}
    res = evaluate_diagnostic_certainty(probs)
    assert isinstance(res, UncertaintyResult)
    assert res.is_borderline is True
    assert res.certainty_level in ["Borderline", "Ambiguous"]
    assert "second opinion" in res.recommendation.lower()


def test_validate_image_quality_valid():
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    # Berikan sedikit variasi intensitas
    arr = np.array(img)
    arr[50:150, 50:150] = 200
    valid_img = Image.fromarray(arr)
    is_valid, msg = validate_image_quality(valid_img)
    assert is_valid is True


def test_validate_image_quality_blank():
    # Gambar polos kosong (tanpa kontras / korup)
    blank_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
    is_valid, msg = validate_image_quality(blank_img)
    assert is_valid is False
    assert "kontras" in msg.lower() or "kosong" in msg.lower()
