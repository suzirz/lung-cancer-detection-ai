"""
Module: Clinical Reader Study & External Benchmark Evaluator
File: src/evaluation/external_benchmark.py

Prinsip Codebase Design:
- Modul Deep: Menyembunyikan kalkulasi metrik studi pembacaan klinis komparatif
  (Unaided vs AI-Aided Radiologist), efisiensi durasi skrining, serta zero-shot validation.
- Interface sederhana: compute_clinical_reader_metrics(...) -> Dict[str, float].
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix


def compute_clinical_reader_metrics(
    y_true: np.ndarray,
    y_pred_unaided: np.ndarray,
    times_unaided: np.ndarray,
    y_pred_aided: np.ndarray,
    times_aided: np.ndarray
) -> Dict[str, float]:
    """
    Menghitung metrik performa komparatif antara radiolog tanpa AI vs radiolog dengan asistensi AI.
    """
    unaided_sens = float(recall_score(y_true, y_pred_unaided, pos_label=1))
    aided_sens = float(recall_score(y_true, y_pred_aided, pos_label=1))

    unaided_acc = float(accuracy_score(y_true, y_pred_unaided))
    aided_acc = float(accuracy_score(y_true, y_pred_aided))

    mean_time_u = float(np.mean(times_unaided))
    mean_time_a = float(np.mean(times_aided))

    sens_gain = float((aided_sens - unaided_sens) * 100.0)
    time_reduction = float((mean_time_u - mean_time_a) / mean_time_u * 100.0) if mean_time_u > 0 else 0.0

    return {
        "unaided_sensitivity": unaided_sens,
        "aided_sensitivity": aided_sens,
        "sensitivity_gain_pct": sens_gain,
        "unaided_accuracy": unaided_acc,
        "aided_accuracy": aided_acc,
        "mean_time_unaided_minutes": mean_time_u,
        "mean_time_aided_minutes": mean_time_a,
        "time_reduction_pct": time_reduction
    }
