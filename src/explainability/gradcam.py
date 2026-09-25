"""
Module: Explainability Layer (Grad-CAM Visualizer)
File: src/explainability/gradcam.py

Prinsip Codebase Design:
- Modul Deep: Menyembunyikan hook gradien konvolusi, aktivasi maju/mundur,
  interpolasi resolusi spasial, dan blending colormap heatmap ke citra asli CT scan.
- Interface sederhana: generate_gradcam(model, image_tensor, target_class) dan overlay_heatmap(orig_img, heatmap).
"""

import os
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) untuk CNN PyTorch.
    Memvisualisasikan area spasial citra yang paling mempengaruhi diagnosis model.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward dan backward hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Menghasilkan peta panas (heatmap) berukuran (H, W) dengan nilai rentang [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        # Global Average Pooling pada gradien saluran
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Bobotkan aktivasi fitur dengan gradien
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        # Rata-rata saluran aktivasi dan terapkan ReLU
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        # Normalisasi ke rentang [0, 1]
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val

        return heatmap


def overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet"
) -> Image.Image:
    """
    Menumpuk heatmap transparan di atas citra CT scan asli.
    """
    # Resize heatmap ke ukuran citra asli
    w, h = original_image.size
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.BILINEAR)
    heatmap_resized = np.array(heatmap_resized) / 255.0

    # Terapkan colormap matplotlib
    cmap = plt.get_cmap(colormap)
    colored_heatmap = cmap(heatmap_resized)[:, :, :3] # Ambil RGB (tanpa alpha)
    colored_heatmap = np.uint8(255 * colored_heatmap)

    orig_arr = np.array(original_image.convert("RGB"))
    blended = np.uint8(orig_arr * (1.0 - alpha) + colored_heatmap * alpha)
    return Image.fromarray(blended)


def explain_prediction(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    original_image: Image.Image,
    target_layer: torch.nn.Module,
    target_class: Optional[int] = None,
    output_path: Optional[str] = None
) -> Tuple[np.ndarray, Image.Image]:
    """
    Interface utama: menghasilkan heatmap dan visualisasi overlay lengkap.
    """
    cam = GradCAM(model, target_layer)
    heatmap = cam.generate_heatmap(image_tensor, target_class=target_class)
    blended_image = overlay_heatmap(original_image, heatmap, alpha=0.45)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        blended_image.save(output_path)

    return heatmap, blended_image
