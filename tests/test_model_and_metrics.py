"""
Unit tests for CNN model architecture and metrics calculation.
File: tests/test_model_and_metrics.py
"""

import os
import torch
import pytest
from src.models.cnn_extractor import build_model
from src.evaluation.metrics import calculate_metrics, plot_confusion_matrix


def test_efficientnet_forward_and_feature_extraction():
    # Model test without preloading heavy weights for rapid testing
    model = build_model(backbone_name="efficientnet_b0", num_classes=3, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)

    # 1. Forward pass (classification logits)
    logits = model(dummy_input)
    assert logits.shape == (2, 3), "Logits harus berbentuk (batch_size, num_classes)"

    # 2. Extract embeddings
    features = model.extract_features(dummy_input)
    assert features.shape == (2, 1280), "Embedding EfficientNet-B0 harus berdimensi 1280"


def test_calculate_metrics_medical_recall():
    y_true = [0, 0, 1, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 1, 2, 2]
    class_names = ["Benign", "Malignant", "Normal"]

    metrics = calculate_metrics(y_true, y_pred, class_names)
    assert "accuracy" in metrics
    assert "recall_macro" in metrics
    assert "recall_Malignant" in metrics
    # Kelas Malignant: 3 benar dari 3 -> Recall 1.0 (100%)
    assert metrics["recall_Malignant"] == 1.0


def test_confusion_matrix_generation(tmp_path):
    y_true = [0, 1, 2, 1]
    y_pred = [0, 1, 1, 1]
    class_names = ["Benign", "Malignant", "Normal"]

    cm_path = os.path.join(tmp_path, "cm_test.png")
    cm = plot_confusion_matrix(y_true, y_pred, class_names, output_path=cm_path)

    assert os.path.exists(cm_path)
    assert cm.shape == (3, 3)
