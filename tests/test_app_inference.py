"""
Unit and integration tests for App Inference Service.
File: tests/test_app_inference.py
"""

import os
import pytest
import numpy as np
from PIL import Image

from app.inference_service import InferenceService, PredictionResult, CLASS_NAMES


MODEL_PATH = "models/baseline_efficientnet_b0_best.pth"


def test_inference_service_initialization():
    """Memverifikasi inisialisasi service dan kesiapan model."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model checkpoint tidak ditemukan di {MODEL_PATH}")

    service = InferenceService(model_path=MODEL_PATH, device="cpu")
    assert service.model is not None
    assert service.model.training is False


def test_inference_service_prediction_dummy():
    """Memverifikasi struktur dan tipe data PredictionResult."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model checkpoint tidak ditemukan di {MODEL_PATH}")

    service = InferenceService(model_path=MODEL_PATH, device="cpu")
    dummy_img = Image.new("RGB", (256, 256), color=(128, 128, 128))

    result = service.predict(dummy_img, alpha=0.5, colormap="jet")

    assert isinstance(result, PredictionResult)
    assert result.class_name in CLASS_NAMES
    assert 0 <= result.class_idx < len(CLASS_NAMES)
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.probabilities) == 3
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-4

    # Cek heatmap dan overlay
    assert isinstance(result.heatmap, np.ndarray)
    assert result.heatmap.ndim == 2
    assert result.overlay_image.size == (256, 256)
    assert result.latency_ms > 0.0


def test_inference_service_samples():
    """Memverifikasi inferensi pada sampel klinis riil di app/samples/."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model checkpoint tidak ditemukan di {MODEL_PATH}")

    sample_files = [
        "app/samples/benign_sample.jpg",
        "app/samples/malignant_sample.jpg",
        "app/samples/normal_sample.jpg",
    ]

    service = InferenceService(model_path=MODEL_PATH, device="cpu")

    for path in sample_files:
        if os.path.exists(path):
            img = Image.open(path)
            res = service.predict(img, alpha=0.45, colormap="jet")
            assert res.class_name in CLASS_NAMES
            assert res.latency_ms < 2000.0  # Harus responsif di bawah 2 detik pada CPU


def test_missing_model_raises_error(tmp_path):
    """Memverifikasi bahwa file model yang tidak ada memicu FileNotFoundError."""
    non_existent = str(tmp_path / "non_existent_weights.pth")
    with pytest.raises(FileNotFoundError):
        InferenceService(model_path=non_existent, device="cpu")
