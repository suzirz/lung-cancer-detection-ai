"""
Module: Feature Extraction Pipeline (CNN Embeddings)
File: src/models/extract_features.py

Prinsip Codebase Design:
- Modul Deep: Menyembunyikan iterasi DataLoader, forward pass ekstraksi tanpa gradien,
  stacking vektor embedding berdimensi 1.280, serta serialisasi terkompresi (.npz).
- Interface sederhana: extract_and_save_embeddings(model, dataloaders, output_dir, device).
"""

import os
from typing import Dict, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.cnn_extractor import build_model


def extract_split_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mengekstrak vektor embedding dan label dari DataLoader tertentu.
    
    Returns:
        X (np.ndarray): Matriks fitur berukuran (N, 1280).
        y (np.ndarray): Vektor label berukuran (N,).
    """
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            # Ekstraksi representasi laten sebelum classification head
            embeddings = model.extract_features(images)
            features_list.append(embeddings.cpu().numpy())
            labels_list.append(labels.numpy())

    X = np.vstack(features_list)
    y = np.concatenate(labels_list)
    return X, y


def extract_and_save_embeddings(
    model_path: str,
    loaders: Dict[str, DataLoader],
    output_dir: str = "data/processed",
    backbone: str = "efficientnet_b0",
    device: torch.device = torch.device("cpu")
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Interface utama: memuat model terlatih, mengekstrak embedding Train/Val/Test,
    dan menyimpannya ke disk dalam format NumPy archive (.npz).
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Memuat bobot model dari: {model_path}")
    
    model = build_model(backbone_name=backbone, num_classes=3, pretrained=False)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)

    data_splits = {}
    for split_name in ("train", "val", "test"):
        if split_name in loaders:
            print(f"[*] Mengekstrak embedding untuk split: {split_name.upper()}...")
            X, y = extract_split_features(model, loaders[split_name], device)
            data_splits[split_name] = (X, y)
            print(f"[OK] {split_name.upper()} diekstrak: Fitur shape = {X.shape}, Label shape = {y.shape}")

    # Simpan ke numpy compressed archive
    out_file = os.path.join(output_dir, f"cnn_features_{backbone}.npz")
    np.savez_compressed(
        out_file,
        X_train=data_splits["train"][0],
        y_train=data_splits["train"][1],
        X_val=data_splits["val"][0],
        y_val=data_splits["val"][1],
        X_test=data_splits["test"][0],
        y_test=data_splits["test"][1]
    )
    print(f"[OK] Seluruh embedding berhasil disimpan di: {out_file}")
    return data_splits
