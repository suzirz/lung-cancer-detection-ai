"""
Module: Hybrid ML Classifier (Random Forest & XGBoost)
File: src/models/ml_classifier.py

Prinsip Codebase Design:
- Modul Deep: Menyembunyikan pemilihan algoritma ensemble (Random Forest / XGBoost),
  penyesuaian class-weighting medis, hyperparameter tuning grid-search, dan serialisasi bobot joblib.
- Interface sederhana: train_hybrid_classifier(X_train, y_train, model_type) dan model.predict(X).
"""

import os
from typing import Dict, Any, Tuple
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from src.evaluation.metrics import calculate_metrics, plot_confusion_matrix


def train_hybrid_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "random_forest",
    tune_hyperparameters: bool = False,
    random_state: int = 42
) -> Any:
    """
    Melatih model Machine Learning klasik di atas representasi fitur embedding CNN.
    """
    model_type = model_type.lower()
    
    if model_type == "random_forest":
        base_clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state
        )
        if tune_hyperparameters:
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [8, 12, 16],
                "min_samples_split": [2, 5]
            }
            grid = GridSearchCV(base_clf, param_grid, cv=3, scoring="recall_macro", n_jobs=-1)
            grid.fit(X_train, y_train)
            print(f"[OK] Best Random Forest Params: {grid.best_params_}")
            return grid.best_estimator_
        else:
            base_clf.fit(X_train, y_train)
            return base_clf

    elif model_type == "xgboost":
        base_clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="mlogloss",
            n_jobs=-1
        )
        if tune_hyperparameters:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.03, 0.1]
            }
            grid = GridSearchCV(base_clf, param_grid, cv=3, scoring="recall_macro", n_jobs=-1)
            grid.fit(X_train, y_train)
            print(f"[OK] Best XGBoost Params: {grid.best_params_}")
            return grid.best_estimator_
        else:
            base_clf.fit(X_train, y_train)
            return base_clf

    else:
        raise ValueError(f"Tipe model '{model_type}' tidak didukung. Pilih 'random_forest' atau 'xgboost'.")


def evaluate_and_save_hybrid(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    model_type: str = "random_forest",
    models_dir: str = "models",
    reports_dir: str = "reports"
) -> Dict[str, float]:
    """
    Mengevaluasi performa model hybrid pada test set dan menyimpan model bobot (.joblib) serta confusion matrix.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test.tolist(), y_pred.tolist(), class_names)

    # Simpan model terkompresi
    model_save_path = os.path.join(models_dir, f"hybrid_{model_type}.joblib")
    joblib.dump(model, model_save_path)
    print(f"[OK] Model Hybrid disimpan di: {model_save_path}")

    # Simpan confusion matrix
    cm_path = os.path.join(reports_dir, f"hybrid_{model_type}_confusion_matrix.png")
    plot_confusion_matrix(
        y_test.tolist(),
        y_pred.tolist(),
        class_names,
        output_path=cm_path,
        title=f"Confusion Matrix Hybrid CNN + {model_type.replace('_', ' ').title()}"
    )
    print(f"[OK] Confusion Matrix disimpan di: {cm_path}")

    return metrics
