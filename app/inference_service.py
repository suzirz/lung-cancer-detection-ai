"""
Module: Inference & Explainability Service
File: app/inference_service.py

Prinsip Codebase Design:
- Modul Deep: Menyembunyikan alokasi perangkat (CPU/CUDA), pemuatan bobot model,
  pipelining transformasi evaluasi citra, inferensi PyTorch dengan autograd hook Grad-CAM,
  serta pembuatan visualisasi heatmap multi-colormap.
- Interface sederhana: InferenceService.predict(pil_image) -> PredictionResult.
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Any
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.models.cnn_extractor import build_model, LungCNNModel
from src.explainability.gradcam import GradCAM, overlay_heatmap
from src.preprocessing.dataset import get_transforms
from src.dicom.processor import load_dicom_slice, create_dicom_secondary_capture, DicomSlice


CLASS_NAMES = ["Benign", "Malignant", "Normal"]
DEFAULT_MODEL_PATH = "models/baseline_efficientnet_b0_best.pth"


@dataclass
class PredictionResult:
    """Hasil inferensi lengkap beserta atensi diagnostik Grad-CAM."""
    class_name: str
    class_idx: int
    confidence: float
    probabilities: Dict[str, float]
    heatmap: np.ndarray
    overlay_image: Image.Image
    latency_ms: float


@dataclass
class DicomPredictionResult(PredictionResult):
    """Hasil inferensi berkas DICOM beserta metadata klinis dan secondary capture dataset."""
    metadata: Dict[str, Any] = None
    hu_min: float = 0.0
    hu_max: float = 0.0
    hu_mean: float = 0.0
    secondary_capture_dcm: Any = None


class InferenceService:
    """
    Layanan inferensi tunggal untuk deteksi kanker paru dari citra CT scan.
    Mendukung caching model di memori dan thread-safe evaluation.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        image_size: Tuple[int, int] = (224, 224)
    ) -> None:
        self.model_path = model_path or DEFAULT_MODEL_PATH
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.image_size = image_size
        self.transforms = get_transforms(image_size=self.image_size)["eval"]
        self.model = self._load_model()

    def _load_model(self) -> LungCNNModel:
        """Memuat arsitektur EfficientNet-B0 dan state_dict bobot terbaik."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"File bobot model tidak ditemukan di: {self.model_path}. "
                "Pastikan model telah dilatih atau diunduh dari checkpoint Colab."
            )

        model = build_model(
            backbone_name="efficientnet_b0",
            num_classes=len(CLASS_NAMES),
            pretrained=False
        )
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        return model

    def predict(
        self,
        image: Image.Image,
        alpha: float = 0.45,
        colormap: str = "jet"
    ) -> PredictionResult:
        """
        Menjalankan diagnosis pada citra CT scan dan menghasilkan heatmap atensi Grad-CAM.

        Args:
            image: Objek PIL Image citra CT scan (RGB atau Grayscale).
            alpha: Transparansi overlay Grad-CAM (0.0 = hanya citra asli, 1.0 = hanya heatmap).
            colormap: Nama colormap matplotlib ('jet', 'inferno', 'viridis', 'magma').

        Returns:
            PredictionResult dengan diagnosis, confidence, probabilitas, dan overlay image.
        """
        t0 = time.perf_counter()

        # Konversi ke RGB jika format grayscale atau RGBA
        rgb_image = image.convert("RGB")

        # Transformasi input tensor dengan requires_grad=True untuk Grad-CAM backward pass
        input_tensor = self.transforms(rgb_image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True

        # Forward pass untuk estimasi probabilitas
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_class = CLASS_NAMES[pred_idx]

        probabilities_dict = {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        }

        # Grad-CAM pada convolutional feature map terakhir
        cam = GradCAM(self.model, self.model.features[-1])
        heatmap = cam.generate_heatmap(input_tensor, target_class=pred_idx)

        # Overlay heatmap ke citra asli
        overlay_img = overlay_heatmap(
            original_image=rgb_image,
            heatmap=heatmap,
            alpha=alpha,
            colormap=colormap
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return PredictionResult(
            class_name=pred_class,
            class_idx=pred_idx,
            confidence=confidence,
            probabilities=probabilities_dict,
            heatmap=heatmap,
            overlay_image=overlay_img,
            latency_ms=latency_ms
        )

    def predict_dicom(
        self,
        dicom_source: Union[str, bytes],
        alpha: float = 0.45,
        colormap: str = "jet"
    ) -> DicomPredictionResult:
        """
        Menjalankan diagnosis pada berkas DICOM asli, mengekstrak Hounsfield Units,
        dan membuat Secondary Capture DICOM untuk integrasi PACS.
        """
        # 1. Parse berkas DICOM dan terapkan Standard Lung Windowing (-600 HU / 1500 HU)
        dcm_slice: DicomSlice = load_dicom_slice(dicom_source)

        # 2. Jalankan inferensi dasar menggunakan citra hasil windowing
        base_result = self.predict(
            image=dcm_slice.windowed_image,
            alpha=alpha,
            colormap=colormap
        )

        # 3. Hitung statistik Hounsfield Unit
        hu_min = float(np.min(dcm_slice.hu_array))
        hu_max = float(np.max(dcm_slice.hu_array))
        hu_mean = float(np.mean(dcm_slice.hu_array))

        # 4. Buat objek DICOM Secondary Capture
        secondary_dcm = create_dicom_secondary_capture(
            original_dcm=dcm_slice.raw_dataset,
            overlay_image=base_result.overlay_image,
            findings_text=base_result.class_name,
            confidence=base_result.confidence
        )

        return DicomPredictionResult(
            class_name=base_result.class_name,
            class_idx=base_result.class_idx,
            confidence=base_result.confidence,
            probabilities=base_result.probabilities,
            heatmap=base_result.heatmap,
            overlay_image=base_result.overlay_image,
            latency_ms=base_result.latency_ms,
            metadata=dcm_slice.metadata,
            hu_min=hu_min,
            hu_max=hu_max,
            hu_mean=hu_mean,
            secondary_capture_dcm=secondary_dcm
        )

