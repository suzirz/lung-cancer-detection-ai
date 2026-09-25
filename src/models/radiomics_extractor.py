"""
Module: Radiomics Feature Extractor & Feature Fusion
File: src/models/radiomics_extractor.py

Prinsip Codebase Design:
- Modul Deep: Menyembunyikan kalkulasi statistik intensitas orde-pertama (first-order statistics),
  analisis tekstur matriks spasial (spatial variance & gradient entropy), serta normalisasi
  dan penggabungan (*feature fusion*) dengan embedding CNN 1.280 dimensi.
- Interface sederhana: RadiomicsExtractor.extract(image) -> np.ndarray dan fuse_features(cnn_emb, rad_feats).
"""

from typing import Union, List, Optional
import numpy as np
from PIL import Image
import cv2
from scipy import stats


RADIOMICS_FEATURE_NAMES = [
    "mean_intensity",
    "std_intensity",
    "variance_intensity",
    "skewness",
    "kurtosis",
    "energy",
    "shannon_entropy",
    "contrast_laplacian_var",
    "sobel_gradient_mean",
    "sobel_gradient_std",
    "percentile_25",
    "percentile_75",
    "interquartile_range",
]


class RadiomicsExtractor:
    """
    Ekstraktor fitur radiomik kuantitatif dari citra CT scan paru (aksial).
    Mencakup statistik intensitas jaringan, morfologi gradien, dan tekstur spasial.
    """

    def __init__(self, target_size: tuple = (224, 224)) -> None:
        self.target_size = target_size
        self.feature_names = RADIOMICS_FEATURE_NAMES

    def extract_from_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Mengekstrak vektor fitur radiomik 1D dari citra tunggal.

        Args:
            image: Objek PIL Image atau NumPy array (grayscale atau RGB).

        Returns:
            np.ndarray 1D dengan panjang sesuai jumlah fitur radiomik.
        """
        # Konversi ke grayscale numpy array
        if isinstance(image, Image.Image):
            img_gray = np.array(image.convert("L"))
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.ndim == 2:
                img_gray = image
            else:
                raise ValueError(f"Dimensi citra tidak didukung: {image.shape}")
        else:
            raise TypeError(f"Tipe input harus PIL Image atau np.ndarray, didapat: {type(image)}")

        # Resize ke target size standar
        if img_gray.shape != self.target_size:
            img_gray = cv2.resize(img_gray, self.target_size, interpolation=cv2.INTER_AREA)

        # Normalisasi ke float [0, 1]
        img_float = img_gray.astype(np.float64) / 255.0
        pixels = img_float.flatten()

        # 1. First-Order Intensity Statistics
        mean_val = float(np.mean(pixels))
        std_val = float(np.std(pixels))
        var_val = float(np.var(pixels))
        if var_val > 1e-10:
            raw_skew = float(stats.skew(pixels))
            raw_kurt = float(stats.kurtosis(pixels))
            skew_val = 0.0 if np.isnan(raw_skew) else raw_skew
            kurt_val = 0.0 if np.isnan(raw_kurt) else raw_kurt
        else:
            skew_val = 0.0
            kurt_val = 0.0
        energy_val = float(np.sum(pixels ** 2)) / len(pixels)

        # Shannon Entropy
        hist, _ = np.histogram(img_gray, bins=64, range=(0, 256), density=True)
        hist = hist[hist > 0]
        entropy_val = float(-np.sum(hist * np.log2(hist)))

        # 2. Texture & Morphological Gradient Features
        # Laplacian variance (indikator ketajaman dan batas nodul spikulasi)
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        laplacian_var = float(laplacian.var())

        # Sobel gradient magnitudes
        sobel_x = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        grad_mean = float(np.mean(grad_mag))
        grad_std = float(np.std(grad_mag))

        # 3. Percentiles & Range
        p25 = float(np.percentile(pixels, 25))
        p75 = float(np.percentile(pixels, 75))
        iqr = p75 - p25

        features = np.array([
            mean_val,
            std_val,
            var_val,
            skew_val,
            kurt_val,
            energy_val,
            entropy_val,
            laplacian_var,
            grad_mean,
            grad_std,
            p25,
            p75,
            iqr,
        ], dtype=np.float32)

        return features

    def batch_extract(self, images: List[Union[Image.Image, np.ndarray]]) -> np.ndarray:
        """Mengekstrak fitur radiomik untuk daftar citra secara berurutan."""
        feats = [self.extract_from_image(img) for img in images]
        return np.vstack(feats)


def fuse_features(
    cnn_features: np.ndarray,
    radiomics_features: np.ndarray,
    normalize_radiomics: bool = True
) -> np.ndarray:
    """
    Menggabungkan vektor representasi CNN (misal 1.280 dimensi) dengan fitur radiomik.

    Args:
        cnn_features: Array bentuk (N, D_cnn) atau (D_cnn,).
        radiomics_features: Array bentuk (N, D_rad) atau (D_rad,).
        normalize_radiomics: Apakah melakukan normalisasi Z-score pada fitur radiomik.

    Returns:
        Array tergabung bentuk (N, D_cnn + D_rad).
    """
    if cnn_features.ndim == 1:
        cnn_features = cnn_features.reshape(1, -1)
    if radiomics_features.ndim == 1:
        radiomics_features = radiomics_features.reshape(1, -1)

    if cnn_features.shape[0] != radiomics_features.shape[0]:
        raise ValueError(
            f"Jumlah sampel tidak cocok: CNN ({cnn_features.shape[0]}) vs "
            f"Radiomics ({radiomics_features.shape[0]})"
        )

    if normalize_radiomics and radiomics_features.shape[0] > 1:
        mean = np.mean(radiomics_features, axis=0, keepdims=True)
        std = np.std(radiomics_features, axis=0, keepdims=True) + 1e-8
        norm_rad = (radiomics_features - mean) / std
    else:
        norm_rad = radiomics_features

    fused = np.concatenate([cnn_features, norm_rad], axis=1)
    return fused
