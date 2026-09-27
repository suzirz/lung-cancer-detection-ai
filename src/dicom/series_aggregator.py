"""
Module: Multi-Slice 3D CT Volume Screening Aggregator
File: src/dicom/series_aggregator.py

Prinsip Codebase Design:
- Menjembatani inferensi 2D per-irisan (slice) menjadi diagnosa komprehensif 3D volume pasien.
- Mengidentifikasi irisan dengan kecurigaan lesi tertinggi (peak malignancy slice)
  serta menyusun narasi ringkasan skrining volume untuk radiolog.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class VolumeSummary:
    """Ringkasan agregasi diagnostik seluruh irisan CT volume pasien."""
    total_slices: int
    overall_diagnosis: str
    suspected_malignant: bool
    peak_slice_index: int
    peak_malignancy_prob: float
    high_risk_slices: List[int] = field(default_factory=list)
    summary_narrative: str = ""


def aggregate_patient_volume(
    slice_predictions: List[Dict[str, Any]],
    malignant_threshold: float = 0.50
) -> VolumeSummary:
    """
    Mengagregasi hasil prediksi dari seluruh irisan (slice) volume CT scan satu pasien.
    Jika satu atau lebih irisan menunjukkan probabilitas ganas >= threshold,
    volume ditandai berisiko tinggi (suspected malignant) dan mengarahkan fokus ke irisan puncak.
    """
    total = len(slice_predictions)
    if total == 0:
        return VolumeSummary(
            total_slices=0,
            overall_diagnosis="Inconclusive",
            suspected_malignant=False,
            peak_slice_index=-1,
            peak_malignancy_prob=0.0,
            high_risk_slices=[],
            summary_narrative="Tidak ada irisan CT scan yang diberikan."
        )

    peak_idx = -1
    peak_prob = -1.0
    high_risk = []

    for pred in slice_predictions:
        s_idx = pred.get("slice_idx", 0)
        probs = pred.get("probabilities", {})
        mal_prob = probs.get("Malignant", 0.0)

        if mal_prob > peak_prob:
            peak_prob = mal_prob
            peak_idx = s_idx

        if mal_prob >= malignant_threshold or pred.get("predicted_class") == "Malignant":
            high_risk.append(s_idx)

    suspected = len(high_risk) > 0

    if suspected:
        overall = "Malignant"
        narrative = (
            f"Terdeteksi kecurigaan lesi kanker paru ganas pada {len(high_risk)} dari {total} irisan. "
            f"Fokus atensi tertinggi berada pada slice #{peak_idx} dengan probabilitas keganasan {peak_prob * 100:.1f}%. "
            "Direkomendasikan evaluasi biopsi atau PET-CT segera."
        )
    else:
        # Cek apakah ada nodul jinak
        benign_slices = [
            pred.get("slice_idx", 0) for pred in slice_predictions
            if pred.get("predicted_class") == "Benign" or pred.get("probabilities", {}).get("Benign", 0) > 0.5
        ]
        if len(benign_slices) > 0:
            overall = "Benign"
            narrative = (
                f"Ditemukan nodul jinak pada {len(benign_slices)} irisan. "
                "Disarankan follow-up CT scan berkala (6-12 bulan)."
            )
        else:
            overall = "Normal"
            narrative = (
                f"Seluruh {total} irisan CT scan dalam batas normal tanpa nodul signifikan."
            )

    return VolumeSummary(
        total_slices=total,
        overall_diagnosis=overall,
        suspected_malignant=suspected,
        peak_slice_index=peak_idx,
        peak_malignancy_prob=peak_prob,
        high_risk_slices=high_risk,
        summary_narrative=narrative
    )
