"""Generate a synthetic clinical DICOM sample with high-attenuation nodule for instant testing in app/samples."""
import io
import os
import numpy as np
from PIL import Image
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, generate_uid

def generate_sample_dicom(output_path="app/samples/clinical_sample.dcm"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows, cols = 224, 224

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientID = "ID-RS-CT-0941"
    ds.PatientName = "ANONYMOUS^PULMOSCAN"
    ds.PatientSex = "M"
    ds.PatientAge = "062Y"
    ds.StudyDescription = "CHEST CT LUNG CANCER SCREENING"
    ds.Modality = "CT"
    ds.SliceThickness = 1.25
    ds.KVP = 120.0
    ds.SeriesNumber = 3
    ds.InstanceNumber = 72

    # HU Rescale
    ds.RescaleIntercept = -1000.0
    ds.RescaleSlope = 1.0

    # Pola CT Scan aksial sintetis:
    # 1. Background rongga tubuh luar (-1000 HU -> raw 0)
    # 2. Dinding dada/tulang (+800 HU -> raw 1800)
    # 3. Parenkim paru (-600 HU -> raw 400)
    # 4. Nodul mencurigakan di lobus kanan (+50 HU -> raw 1050)
    arr = np.zeros((rows, cols), dtype=np.uint16)

    # Lingkaran rongga dada
    y, x = np.ogrid[:rows, :cols]
    chest_mask = ((x - 112)**2 + (y - 112)**2) <= 90**2
    arr[chest_mask] = 1300  # Soft tissue

    # Paru kiri dan kanan (-600 HU -> raw 400)
    left_lung_mask = (((x - 75)**2) / 25**2 + ((y - 112)**2) / 55**2) <= 1
    right_lung_mask = (((x - 149)**2) / 25**2 + ((y - 112)**2) / 55**2) <= 1
    arr[left_lung_mask] = 400
    arr[right_lung_mask] = 400

    # Nodul mencurigakan (+50 HU -> raw 1050) di dalam paru kanan
    nodule_mask = ((x - 148)**2 + (y - 105)**2) <= 9**2
    arr[nodule_mask] = 1050

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

    ds.save_as(output_path, enforce_file_format=True)
    print(f"Sample clinical DICOM saved to: {output_path}")

if __name__ == "__main__":
    generate_sample_dicom()
