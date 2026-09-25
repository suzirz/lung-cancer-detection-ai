"""
Unit & Integration Test for Preprocessing Pipeline.
File: tests/test_preprocessing.py

Memverifikasi fungsi scan_dataset, stratified splitting, dan PyTorch DataLoader.
"""

import os
import json
import pytest
import torch
from src.preprocessing.dataset import scan_dataset, get_transforms, create_dataloaders, LungCTDataset
from src.preprocessing.split import split_dataset


def test_get_transforms():
    tfms = get_transforms(image_size=(224, 224))
    assert "train" in tfms
    assert "eval" in tfms


def test_split_dataset_ratio_and_stratification():
    # Buat sampel dummy: 100 kelas 0, 100 kelas 1, 100 kelas 2
    dummy_samples = [("dummy_path.jpg", i % 3) for i in range(300)]
    splits = split_dataset(dummy_samples, val_ratio=0.10, test_ratio=0.10, seed=42)
    
    assert len(splits["train"]) == 240
    assert len(splits["val"]) == 30
    assert len(splits["test"]) == 30

    # Cek sebaran kelas di test set (harus seimbang 10 tiap kelas)
    test_labels = [s[1] for s in splits["test"]]
    assert test_labels.count(0) == 10
    assert test_labels.count(1) == 10
    assert test_labels.count(2) == 10


def test_split_summary_integrity():
    summary_path = "data/splits/split_summary.json"
    assert os.path.exists(summary_path), "File split_summary.json harus sudah ada"
    
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    assert summary["total_samples"] == 12184
    assert summary["split_counts"]["train"] + summary["split_counts"]["val"] + summary["split_counts"]["test"] == 12184


def test_dataloader_batch_generation():
    # Ambil 10 sampel nyata dari Train split
    with open("data/splits/train_split.json", "r", encoding="utf-8") as f:
        train_samples = json.load(f)[:10]
        
    loaders, _ = create_dataloaders(
        train_samples=train_samples,
        val_samples=train_samples[:2],
        test_samples=train_samples[:2],
        batch_size=4,
        image_size=(224, 224),
        num_workers=0, # 0 untuk testing lokal cepat
        pin_memory=False
    )
    
    train_loader = loaders["train"]
    images, labels = next(iter(train_loader))
    
    assert images.shape == (4, 3, 224, 224)
    assert labels.shape == (4,)
    assert images.dtype == torch.float32
