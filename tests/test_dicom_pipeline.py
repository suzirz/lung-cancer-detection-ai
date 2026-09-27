"""
Unit Tests: Clinical DICOM Ingestion, Hounsfield Units, & Secondary Capture Export
File: tests/test_dicom_pipeline.py
"""

import io
import os
import pytest
import numpy as np
from PIL import Image
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, generate_uid

from src.dicom.processor import (
    get_hounsfield_units,
    apply_lung_window,
    load_dicom_slice,
    create_dicom_secondary_capture,
    DicomSlice
)
from app.inference_service import InferenceService, DicomPredictionResult


def create_synthetic_dicom_file(rows: int = 128, cols: int = 128) -> bytes:
    """Helper untuk membuat file DICOM sintetis CT scan dengan skala HU."""
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientID = "PATIENT_LUNG_99"
    ds.PatientSex = "M"
    ds.PatientAge = "058Y"
    ds.StudyDescription = "CT CHEST LUNG SCREENING"
    ds.Modality = "CT"
    ds.SliceThickness = 1.25
    ds.KVP = 120.0
    ds.SeriesNumber = 2
    ds.InstanceNumber = 45

    # CT Rescale attributes: Raw 0 -> -1000 HU (Air), Raw 1000 -> 0 HU (Water), Raw 2000 -> +1000 HU (Bone)
    ds.RescaleIntercept = -1000.0
    ds.RescaleSlope = 1.0

    # Buat array piksel sintetis
    # Simulasikan parenkim paru (-600 HU => raw 400) dan nodul densitas tinggi (-100 HU => raw 900)
    arr = np.full((rows, cols), 400, dtype=np.uint16)
    arr[40:80, 40:80] = 900  # Area nodul

    ds.Rows, ds.Columns = rows, cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def test_hounsfield_units_conversion():
    dcm_bytes = create_synthetic_dicom_file(64, 64)
    slice_obj = load_dicom_slice(dcm_bytes)

    assert isinstance(slice_obj, DicomSlice)
    # Background 400 + (-1000) = -600 HU
    # Nodul 900 + (-1000) = -100 HU
    assert np.isclose(slice_obj.hu_array[0, 0], -600.0, atol=1.0)
    assert np.isclose(slice_obj.hu_array[50, 50], -100.0, atol=1.0)
    assert slice_obj.metadata["patient_id"] == "PATIENT_LUNG_99"
    assert slice_obj.metadata["slice_thickness_mm"] == 1.25


def test_lung_windowing():
    hu_test = np.array([[-1500.0, -600.0], [0.0, 500.0]], dtype=np.float32)
    # Default Lung Window: [-1350, +150]
    img = apply_lung_window(hu_test, window_width=1500.0, window_level=-600.0)

    assert isinstance(img, Image.Image)
    assert img.size == (2, 2)
    arr = np.array(img)
    # Nilai di bawah -1350 menjadi 0
    assert arr[0, 0, 0] == 0
    # Nilai di atas 150 menjadi 255
    assert arr[1, 1, 0] == 255


def test_dicom_secondary_capture_creation():
    dcm_bytes = create_synthetic_dicom_file(64, 64)
    dcm = pydicom.dcmread(io.BytesIO(dcm_bytes))
    mock_overlay = Image.new("RGB", (64, 64), color=(255, 0, 0))

    sc = create_dicom_secondary_capture(
        original_dcm=dcm,
        overlay_image=mock_overlay,
        findings_text="Malignant",
        confidence=0.985
    )

    assert isinstance(sc, Dataset)
    assert sc.Modality == "OT"
    assert sc.PatientID == "PATIENT_LUNG_99"
    assert "PulmoScan AI Triage" in sc.SeriesDescription
    assert sc.SamplesPerPixel == 3  # RGB image
    assert sc.Rows == 64
    assert sc.Columns == 64


def test_inference_service_predict_dicom(tmp_path):
    # Buat dummy checkpoint model jika belum ada
    model_path = str(tmp_path / "mock_model.pth")
    from src.models.cnn_extractor import build_model
    import torch
    dummy_model = build_model("efficientnet_b0", num_classes=3, pretrained=False)
    torch.save(dummy_model.state_dict(), model_path)

    service = InferenceService(model_path=model_path, device="cpu")
    dcm_bytes = create_synthetic_dicom_file(128, 128)

    result = service.predict_dicom(dcm_bytes)
    assert isinstance(result, DicomPredictionResult)
    assert result.class_name in ["Benign", "Malignant", "Normal"]
    assert result.metadata["patient_id"] == "PATIENT_LUNG_99"
    assert result.secondary_capture_dcm is not None
    assert result.hu_min < result.hu_max
