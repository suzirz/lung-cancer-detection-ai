"""
Module: Evaluation Metrics & Diagnostic Reporting
File: src/evaluation/metrics.py

Prinsip Codebase Design:
- Modul Deep: Menyembunyikan kalkulasi multi-class confusion matrix, sensitivity/recall per kelas,
  specificity, macro/weighted F1, serta rendering plot PNG visual.
- Prioritas Utama Sesuai PRD: Recall / Sensitivity medis.
"""

import os
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score, accuracy_score


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str]
) -> Dict[str, float]:
    """
    Menghitung seluruh metrik diagnostik medis utama.
    """
    acc = accuracy_score(y_true, y_pred)
    # Recall per kelas & macro
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Recall spesifik per kelas
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "recall_macro": float(recall_macro),
        "recall_weighted": float(recall_weighted),
        "precision_macro": float(precision_macro),
        "f1_macro": float(f1_macro),
    }

    for idx, name in enumerate(class_names):
        metrics[f"recall_{name}"] = float(recall_per_class[idx])

    return metrics


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    output_path: str = "reports/baseline_confusion_matrix.png",
    title: str = "Confusion Matrix Diagnosa Kanker Paru"
) -> np.ndarray:
    """
    Menghasilkan dan menyimpan visualisasi Confusion Matrix.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="Ground Truth (Label Asli)",
        xlabel="Prediksi Model"
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Tampilkan angka di setiap sel
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return cm
