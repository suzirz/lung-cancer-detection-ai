"""
Module: Diagnostic Uncertainty & Borderline Safeguards
File: src/evaluation/uncertainty.py

Prinsip Codebase Design:
- Modul Deep: Menyembunyikan kalkulasi Shannon entropy, margin probabilitas,
  evaluasi kualitas citra input (kontras/noise), serta logika rekomendasi klinis.
- Menjamin AI tidak memberikan diagnosis over-confident pada kasus batas (borderline).
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from PIL import Image


@dataclass
class UncertaintyResult:
    """Hasil evaluasi kepastian diagnostik medis."""
    is_borderline: bool
    certainty_level: str  # "High", "Moderate", "Borderline", "Ambiguous"
    entropy: float
    confidence: float
    recommendation: str


def compute_shannon_entropy(probabilities: Dict[str, float]) -> float:
    """Menghitung Shannon entropy berbasis natural log (e) dari distribusi probabilitas."""
    probs = np.array(list(probabilities.values()), dtype=np.float64)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def evaluate_diagnostic_certainty(
    probabilities: Dict[str, float],
    margin_threshold: float = 0.15,
    entropy_threshold: float = 0.65
) -> UncertaintyResult:
    """
    Mengevaluasi tingkat kepastian diagnosa model AI.
    Jika selisih dua probabilitas teratas < margin_threshold atau entropy tinggi,
    kasus diklasifikasikan sebagai borderline/ambigu dan memicu rekomendasi second opinion.
    """
    sorted_probs = sorted(probabilities.values(), reverse=True)
    top1 = sorted_probs[0] if len(sorted_probs) > 0 else 1.0
    top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0

    margin = top1 - top2
    entropy = compute_shannon_entropy(probabilities)

    if margin < margin_threshold or entropy >= entropy_threshold:
        is_borderline = True
        certainty_level = "Borderline" if margin < margin_threshold else "Ambiguous"
        recommendation = (
            "Kasus berada pada ambang batas klinis (borderline). "
            "Disarankan verifikasi manual (second opinion) oleh dokter spesialis radiologi "
            "atau pemeriksaan CT scan lanjutan."
        )
    elif top1 >= 0.85:
        is_borderline = False
        certainty_level = "High"
        recommendation = "Tingkat kepastian model tinggi. Temuan konsisten dengan pola patologi tipikal."
    else:
        is_borderline = False
        certainty_level = "Moderate"
        recommendation = "Tingkat kepastian moderat. Evaluasi klinis pendukung dianjurkan."

    return UncertaintyResult(
        is_borderline=is_borderline,
        certainty_level=certainty_level,
        entropy=entropy,
        confidence=top1,
        recommendation=recommendation
    )


def validate_image_quality(image: Image.Image, min_contrast_std: float = 5.0) -> Tuple[bool, str]:
    """
    Memvalidasi apakah citra yang diunggah memiliki kualitas layak untuk analisis CT scan.
    Mendeteksi citra polos/blank, korup, atau tanpa kontras anatomi.
    """
    if image is None:
        return False, "Citra kosong atau gagal dimuat."

    arr = np.array(image.convert("L"), dtype=np.float32)
    std_val = float(np.std(arr))

    if std_val < min_contrast_std:
        return False, f"Citra tidak memiliki kontras yang memadai (std={std_val:.1f} < {min_contrast_std}). Berkas mungkin gambar polos kosong atau korup."

    # Periksa dimensi minimum
    w, h = image.size
    if w < 64 or h < 64:
        return False, f"Dimensi citra terlalu kecil ({w}x{h} px). Standar CT scan minimal 64x64 piksel."

    return True, "Kualitas citra valid untuk inferensi."
