"""
Unit tests for Radiomics Feature Extractor & Feature Fusion.
File: tests/test_radiomics.py
"""

import numpy as np
import pytest
from PIL import Image

from src.models.radiomics_extractor import RadiomicsExtractor, fuse_features, RADIOMICS_FEATURE_NAMES


def test_radiomics_extractor_single_pil_image():
    """Menguji ekstraksi fitur radiomik dari objek PIL Image."""
    extractor = RadiomicsExtractor(target_size=(224, 224))
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))

    features = extractor.extract_from_image(img)

    assert isinstance(features, np.ndarray)
    assert features.shape == (len(RADIOMICS_FEATURE_NAMES),)
    assert not np.any(np.isnan(features))
    assert not np.any(np.isinf(features))


def test_radiomics_extractor_numpy_grayscale():
    """Menguji ekstraksi dari 2D grayscale numpy array."""
    extractor = RadiomicsExtractor(target_size=(224, 224))
    arr = np.random.randint(0, 256, (128, 128), dtype=np.uint8)

    features = extractor.extract_from_image(arr)

    assert features.shape == (len(RADIOMICS_FEATURE_NAMES),)
    assert not np.any(np.isnan(features))


def test_radiomics_batch_extraction():
    """Menguji ekstraksi batch citra."""
    extractor = RadiomicsExtractor(target_size=(224, 224))
    images = [
        Image.new("L", (100, 100), color=50),
        Image.new("L", (100, 100), color=150),
        Image.new("L", (100, 100), color=220),
    ]

    batch_feats = extractor.batch_extract(images)

    assert batch_feats.shape == (3, len(RADIOMICS_FEATURE_NAMES))
    assert not np.any(np.isnan(batch_feats))


def test_fuse_features_dimensions():
    """Menguji fusi embedding CNN (1280 dim) dengan radiomik (13 dim)."""
    n_samples = 5
    cnn_dim = 1280
    rad_dim = len(RADIOMICS_FEATURE_NAMES)

    cnn_features = np.random.randn(n_samples, cnn_dim).astype(np.float32)
    rad_features = np.random.randn(n_samples, rad_dim).astype(np.float32)

    fused = fuse_features(cnn_features, rad_features, normalize_radiomics=True)

    assert fused.shape == (n_samples, cnn_dim + rad_dim)
    assert not np.any(np.isnan(fused))


def test_fuse_features_mismatched_samples_raises():
    """Menguji proteksi terhadap dimensi sampel yang tidak serasi."""
    cnn_features = np.random.randn(4, 1280)
    rad_features = np.random.randn(5, 13)

    with pytest.raises(ValueError, match="Jumlah sampel tidak cocok"):
        fuse_features(cnn_features, rad_features)
