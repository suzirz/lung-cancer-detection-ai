"""
Unit tests for Explainability (Grad-CAM) and Final Evaluation (ROC-AUC & CV).
File: tests/test_explainability_and_final_eval.py
"""

import os
import numpy as np
import torch
import pytest
from PIL import Image

from src.models.cnn_extractor import build_model
from src.explainability.gradcam import GradCAM, overlay_heatmap, explain_prediction
from src.evaluation.final_eval import compute_roc_auc_multiclass, run_cross_validation_audit


def test_gradcam_heatmap_dimensions():
    model = build_model(backbone_name="efficientnet_b0", num_classes=3, pretrained=False)
    # Target convolutional layer: layer konvolusi terakhir pada features
    target_layer = model.features[-1]

    cam = GradCAM(model, target_layer)
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)

    heatmap = cam.generate_heatmap(dummy_input, target_class=1)
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2
    assert np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0)


def test_overlay_heatmap():
    dummy_orig = Image.new("RGB", (224, 224), color=(100, 100, 100))
    dummy_heatmap = np.random.rand(7, 7)

    blended = overlay_heatmap(dummy_orig, dummy_heatmap, alpha=0.5)
    assert blended.size == (224, 224)
    assert blended.mode == "RGB"


def test_roc_auc_multiclass(tmp_path):
    y_true = [0, 1, 2, 0, 1, 2, 1, 0, 2, 1]
    # Simulasi probabilitas prediksi
    y_probs = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.1, 0.8],
        [0.05, 0.9, 0.05],
        [0.85, 0.1, 0.05],
        [0.1, 0.1, 0.8],
        [0.15, 0.8, 0.05]
    ])
    class_names = ["Benign", "Malignant", "Normal"]
    out_path = str(tmp_path / "roc_test.png")

    roc_dict = compute_roc_auc_multiclass(y_true, y_probs, class_names, output_path=out_path)
    assert "macro_auc" in roc_dict
    assert roc_dict["macro_auc"] >= 0.90
    assert os.path.exists(out_path)


def test_cross_validation_audit():
    np.random.seed(42)
    X = np.random.randn(90, 16)
    y = np.array([0]*30 + [1]*30 + [2]*30)

    cv_results = run_cross_validation_audit(X, y, n_splits=3)
    assert "mean_recall_macro" in cv_results
    assert "mean_accuracy" in cv_results
    assert cv_results["cv_folds"] == 3
