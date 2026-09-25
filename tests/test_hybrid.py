"""
Unit tests for Feature Extraction and Hybrid ML Classifiers.
File: tests/test_hybrid.py
"""

import numpy as np
import pytest
from src.models.ml_classifier import train_hybrid_classifier, evaluate_and_save_hybrid


def test_train_random_forest_on_embeddings():
    # Simulasi 60 sampel embedding (20 per kelas), dimensi 64
    np.random.seed(42)
    X = np.random.randn(60, 64)
    y = np.array([0]*20 + [1]*20 + [2]*20)

    model = train_hybrid_classifier(X, y, model_type="random_forest", tune_hyperparameters=False)
    preds = model.predict(X)

    assert len(preds) == 60
    assert set(preds).issubset({0, 1, 2})


def test_train_xgboost_on_embeddings():
    np.random.seed(42)
    X = np.random.randn(60, 64)
    y = np.array([0]*20 + [1]*20 + [2]*20)

    model = train_hybrid_classifier(X, y, model_type="xgboost", tune_hyperparameters=False)
    preds = model.predict(X)

    assert len(preds) == 60
    assert set(preds).issubset({0, 1, 2})


def test_evaluate_and_save_hybrid(tmp_path):
    np.random.seed(42)
    X_train = np.random.randn(30, 32)
    y_train = np.array([0]*10 + [1]*10 + [2]*10)
    X_test = np.random.randn(15, 32)
    y_test = np.array([0]*5 + [1]*5 + [2]*5)

    model = train_hybrid_classifier(X_train, y_train, model_type="random_forest")
    metrics = evaluate_and_save_hybrid(
        model=model,
        X_test=X_test,
        y_test=y_test,
        class_names=["Benign", "Malignant", "Normal"],
        model_type="random_forest",
        models_dir=str(tmp_path / "models"),
        reports_dir=str(tmp_path / "reports")
    )

    assert "accuracy" in metrics
    assert "recall_macro" in metrics
    assert (tmp_path / "models" / "hybrid_random_forest.joblib").exists()
    assert (tmp_path / "reports" / "hybrid_random_forest_confusion_matrix.png").exists()
