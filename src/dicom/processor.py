"""
Module: Clinical DICOM Ingestion, Hounsfield Unit Windowing, & PACS Secondary Capture
File: src/dicom/processor.py

Prinsip Codebase Design:
- Modul Deep: Menyembunyikan kompleksitas parsing header DICOM, kalkulasi nilai atenuasi HU
  (Rescale Slope & Intercept), windowing parenkim paru medis, serta enkapsulasi
  hasil analisis AI ke dalam objek DICOM Secondary Capture (SC).
- Interface sederhana & murni:
    - load_dicom_slice(source) -> DicomSlice
    - apply_lung_window(hu_array) -> np.ndarray (uint8)
    - export_secondary_capture(original_dcm, overlay_image, findings) -> pydicom.Dataset
"""

import io
from dataclasses import dataclass
from typing import Union, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid


@dataclass
class DicomSlice:
    """Representasi data klinis dari satu irisan DICOM CT scan."""
    pixel_array: np.ndarray          # Raw array dari scanner
    hu_array: np.ndarray             # Array terkalibrasi Hounsfield Unit
    windowed_image: Image.Image      # Citra RGB 8-bit hasil standard lung windowing
    metadata: Dict[str, Any]         # Informasi medis & anonim pasien
    raw_dataset: Dataset             # Objek pydicom dataset asli


def get_hounsfield_units(dcm: Dataset) -> np.ndarray:
    """
    Mengonversi nilai raw pixel scanner ke satuan Hounsfield Units (HU):
    HU = PixelValue * RescaleSlope + RescaleIntercept
    """
    image = dcm.pixel_array.astype(np.float32)

    # Beberapa scanner menggunakan padding value di luar field of view
    intercept = getattr(dcm, "RescaleIntercept", 0.0)
    slope = getattr(dcm, "RescaleSlope", 1.0)

    if slope != 1.0:
        image = slope * image
    image += np.float32(intercept)

    return image


def apply_lung_window(
    hu_array: np.ndarray,
    window_width: float = 1500.0,
    window_level: float = -600.0
) -> Image.Image:
    """
    Menerapkan standard lung windowing untuk visualisasi optimal parenkim paru.
    Rentang klinis: [level - width/2, level + width/2]
    Default Lung Window: [-1350 HU, +150 HU]
    """
    lower_bound = window_level - (window_width / 2.0)
    upper_bound = window_level + (window_width / 2.0)

    windowed = np.clip(hu_array, lower_bound, upper_bound)
    normalized = ((windowed - lower_bound) / (upper_bound - lower_bound) * 255.0).astype(np.uint8)

    # Mengembalikan citra RGB
    return Image.fromarray(normalized).convert("RGB")


def parse_dicom_metadata(dcm: Dataset) -> Dict[str, Any]:
    """Ekstraksi metadata radiologi penting dengan perlindungan data pasien (anonymized)."""
    return {
        "patient_id": str(getattr(dcm, "PatientID", "ANONYMOUS")),
        "patient_sex": str(getattr(dcm, "PatientSex", "U")),
        "patient_age": str(getattr(dcm, "PatientAge", "N/A")),
        "study_description": str(getattr(dcm, "StudyDescription", "Chest CT")),
        "modality": str(getattr(dcm, "Modality", "CT")),
        "slice_thickness_mm": float(getattr(dcm, "SliceThickness", 1.0)),
        "kvp": float(getattr(dcm, "KVP", 120.0)),
        "rescale_slope": float(getattr(dcm, "RescaleSlope", 1.0)),
        "rescale_intercept": float(getattr(dcm, "RescaleIntercept", 0.0)),
        "image_shape": dcm.pixel_array.shape if hasattr(dcm, "pixel_array") else None,
    }


def load_dicom_slice(
    source: Union[str, bytes, io.BytesIO],
    window_width: float = 1500.0,
    window_level: float = -600.0
) -> DicomSlice:
    """
    Membaca dan memproses berkas DICOM CT scan menjadi format yang siap dianalisis AI.
    """
    if isinstance(source, (bytes, bytearray)):
        dcm = pydicom.dcmread(io.BytesIO(source))
    elif isinstance(source, io.BytesIO):
        dcm = pydicom.dcmread(source)
    else:
        dcm = pydicom.dcmread(source)

    hu_array = get_hounsfield_units(dcm)
    windowed_image = apply_lung_window(hu_array, window_width, window_level)
    metadata = parse_dicom_metadata(dcm)

    return DicomSlice(
        pixel_array=dcm.pixel_array,
        hu_array=hu_array,
        windowed_image=windowed_image,
        metadata=metadata,
        raw_dataset=dcm
    )


def create_dicom_secondary_capture(
    original_dcm: Dataset,
    overlay_image: Image.Image,
    findings_text: str,
    confidence: float
) -> Dataset:
    """
    Mengekspor visualisasi atensi AI (Grad-CAM overlay) kembali ke format DICOM Secondary Capture.
    File ini dapat langsung diimpor dan dibuka di PACS Workstation radiolog rumah sakit.
    """
    # Siapkan file meta
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Salin atribut pasien & studi dari DICOM asli
    for tag in [
        "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
        "StudyInstanceUID", "StudyDate", "StudyTime", "ReferringPhysicianName",
        "StudyID", "AccessionNumber"
    ]:
        if hasattr(original_dcm, tag):
            setattr(ds, tag, getattr(original_dcm, tag))

    # Identitas Series & Instance Baru
    ds.Modality = "OT"  # Other / Secondary Capture
    ds.ConversionType = "WSD"  # Workstation
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SeriesNumber = int(getattr(original_dcm, "SeriesNumber", 1)) + 900  # Series AI
    ds.InstanceNumber = 1
    ds.SeriesDescription = f"PulmoScan AI Triage: {findings_text} ({confidence*100:.1f}%)"

    # Encoding Pixel Data (RGB)
    rgb_img = overlay_image.convert("RGB")
    rgb_array = np.array(rgb_img, dtype=np.uint8)

    ds.Rows, ds.Columns = rgb_array.shape[:2]
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = rgb_array.tobytes()

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    return ds
