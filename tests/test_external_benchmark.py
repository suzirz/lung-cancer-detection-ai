"""
Unit tests for External Benchmark & Reader Study Protocol.
File: tests/test_external_benchmark.py
"""

import pytest
import numpy as np

from src.evaluation.external_benchmark import compute_clinical_reader_metrics


def test_clinical_reader_metrics():
    # Simulasi 50 kasus pasien yang dibaca oleh 2 radiolog (dengan vs tanpa AI assist)
    y_true = np.array([1]*25 + [0]*25)
    # Radiolog tanpa AI: Sensitivitas 84%, Waktu rata-rata 4.2 menit/kasus
    y_pred_manual = np.array([1]*21 + [0]*4 + [0]*22 + [1]*3)
    reading_times_manual = np.full(50, 4.2)

    # Radiolog dengan PulmoScan AI: Sensitivitas 96%, Waktu rata-rata 2.1 menit/kasus
    y_pred_ai_assisted = np.array([1]*24 + [0]*1 + [0]*23 + [1]*2)
    reading_times_ai = np.full(50, 2.1)

    metrics = compute_clinical_reader_metrics(
        y_true=y_true,
        y_pred_unaided=y_pred_manual,
        times_unaided=reading_times_manual,
        y_pred_aided=y_pred_ai_assisted,
        times_aided=reading_times_ai
    )

    assert metrics["sensitivity_gain_pct"] > 10.0
    assert metrics["time_reduction_pct"] >= 50.0
    assert metrics["unaided_sensitivity"] == 0.84
    assert metrics["aided_sensitivity"] == 0.96
