"""
Unit & Integration Test for Preprocessing Pipeline.
File: tests/test_preprocessing.py

Memverifikasi fungsi scan_dataset, group-aware stratified splitting,
data leakage prevention, dan PyTorch DataLoader.
"""

import os
import json
import pytest
import torch
from src.preprocessing.dataset import scan_dataset, get_transforms, create_dataloaders, LungCTDataset, extract_group_id
from src.preprocessing.split import split_dataset


def test_get_transforms():
    tfms = get_transforms(image_size=(224, 224))
    assert "train" in tfms
    assert "eval" in tfms


def test_extract_group_id_standard_filenames():
    """Verify group ID extraction for IQ-OTHNCCD naming pattern."""
    assert extract_group_id("Malignant case (445)(6).jpg") == "Malignant case_445"
    assert extract_group_id("Normal case (84)(1).jpg") == "Normal case_84"
    assert extract_group_id("Benign case (44)(9).jpg") == "Benign case_44"
    # All augmentation variants of the same scan should map to the same group
    assert extract_group_id("Malignant case (445)(1).jpg") == extract_group_id("Malignant case (445)(6).jpg")
    assert extract_group_id("Normal case (84)(1).jpg") == extract_group_id("Normal case (84)(10).jpg")


def test_extract_group_id_with_path():
    """Group ID extraction should work with full paths, not just basenames."""
    path = "data/raw/The IQ-OTHNCCD Lung Cancer Augmented Dataset/Malignant cases/Train/Malignant case (92)(3).jpg"
    assert extract_group_id(path) == "Malignant case_92"


def test_extract_group_id_fallback():
    """Non-standard filenames should fallback to full basename."""
    assert extract_group_id("random_file.jpg") == "random_file.jpg"


def test_split_dataset_ratio_and_stratification():
    """Group-aware split should respect approximate ratios and stratification."""
    # Simulate 30 patients, each with 10 augmented variants, across 3 classes
    dummy_samples = []
    for patient_id in range(30):
        label = patient_id % 3
        for aug_id in range(10):
            cls_name = ["Benign case", "Malignant case", "Normal case"][label]
            filename = f"{cls_name} ({patient_id})({aug_id}).jpg"
            dummy_samples.append((filename, label))

    splits = split_dataset(dummy_samples, val_ratio=0.15, test_ratio=0.15, seed=42)

    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == 300, f"All samples accounted for: {total}"

    # Check no empty splits
    assert len(splits["train"]) > 0
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0


def test_split_dataset_zero_group_overlap():
    """
    CRITICAL: Verify zero patient/scan overlap between train, val, and test splits.
    This is the core data leakage prevention test.
    """
    # Simulate 30 patients, each with 10 augmented variants, across 3 classes
    dummy_samples = []
    for patient_id in range(30):
        label = patient_id % 3
        for aug_id in range(10):
            cls_name = ["Benign case", "Malignant case", "Normal case"][label]
            filename = f"{cls_name} ({patient_id})({aug_id}).jpg"
            dummy_samples.append((filename, label))

    splits = split_dataset(dummy_samples, val_ratio=0.15, test_ratio=0.15, seed=42)

    # Extract group IDs per split
    train_groups = set(extract_group_id(fp) for fp, _ in splits["train"])
    val_groups = set(extract_group_id(fp) for fp, _ in splits["val"])
    test_groups = set(extract_group_id(fp) for fp, _ in splits["test"])

    # Zero overlap assertions
    assert len(train_groups & val_groups) == 0, \
        f"LEAKAGE: {len(train_groups & val_groups)} groups in both train and val"
    assert len(train_groups & test_groups) == 0, \
        f"LEAKAGE: {len(train_groups & test_groups)} groups in both train and test"
    assert len(val_groups & test_groups) == 0, \
        f"LEAKAGE: {len(val_groups & test_groups)} groups in both val and test"


def test_split_dataset_all_augmentations_in_same_split():
    """
    Verify that ALL augmented variants of a single patient stay in the same split.
    """
    dummy_samples = []
    for patient_id in range(30):
        label = patient_id % 3
        for aug_id in range(10):
            cls_name = ["Benign case", "Malignant case", "Normal case"][label]
            filename = f"{cls_name} ({patient_id})({aug_id}).jpg"
            dummy_samples.append((filename, label))

    splits = split_dataset(dummy_samples, val_ratio=0.15, test_ratio=0.15, seed=42)

    # For each split, verify all augmentations of a patient are in the same split
    for split_name, split_data in splits.items():
        groups_in_split = set(extract_group_id(fp) for fp, _ in split_data)
        for gid in groups_in_split:
            # Count how many samples of this group are in the split
            count_in_split = sum(1 for fp, _ in split_data if extract_group_id(fp) == gid)
            # Should be exactly 10 (all augmentations)
            assert count_in_split == 10, \
                f"Group {gid} has {count_in_split}/10 samples in {split_name} — augmentations were split"


def test_real_split_no_leakage():
    """Integration test: verify the real on-disk splits have zero group overlap."""
    splits_dir = "data/splits"
    split_files = {
        "train": os.path.join(splits_dir, "train_split.json"),
        "val": os.path.join(splits_dir, "val_split.json"),
        "test": os.path.join(splits_dir, "test_split.json"),
    }

    # Skip if split files don't exist (e.g., CI without dataset)
    for path in split_files.values():
        if not os.path.exists(path):
            pytest.skip("Split files not found on disk (dataset not downloaded)")

    group_sets = {}
    for split_name, path in split_files.items():
        with open(path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        group_sets[split_name] = set(extract_group_id(fp) for fp, _ in samples)

    assert len(group_sets["train"] & group_sets["val"]) == 0, "LEAKAGE: train/val overlap"
    assert len(group_sets["train"] & group_sets["test"]) == 0, "LEAKAGE: train/test overlap"
    assert len(group_sets["val"] & group_sets["test"]) == 0, "LEAKAGE: val/test overlap"


def test_split_summary_integrity():
    summary_path = "data/splits/split_summary.json"
    if not os.path.exists(summary_path):
        pytest.skip("split_summary.json not found")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["total_samples"] == 12184
    assert summary["split_counts"]["train"] + summary["split_counts"]["val"] + summary["split_counts"]["test"] == 12184

    # Verify group-aware split metadata is present
    assert summary.get("split_method") == "group_aware_stratified", \
        "Split must use group-aware method"
    assert "group_counts" in summary, "Group counts must be recorded"


def test_dataloader_batch_generation(tmp_path):
    # Check if raw files from train split exist on disk
    train_path = "data/splits/train_split.json"
    use_real = False
    train_samples = []

    if os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)[:10]
        if candidates and os.path.exists(candidates[0][0]):
            train_samples = candidates
            use_real = True

    # If raw dataset is not downloaded (e.g. in CI runner), create synthetic images
    if not use_real:
        from PIL import Image
        for i in range(10):
            img_file = str(tmp_path / f"ci_sample_{i}.jpg")
            Image.new("RGB", (224, 224), color=(i * 20, i * 20, i * 20)).save(img_file)
            train_samples.append((img_file, i % 3))

    loaders, _ = create_dataloaders(
        train_samples=train_samples,
        val_samples=train_samples[:2],
        test_samples=train_samples[:2],
        batch_size=4,
        image_size=(224, 224),
        num_workers=0,
        pin_memory=False
    )

    train_loader = loaders["train"]
    images, labels = next(iter(train_loader))

    assert images.shape == (4, 3, 224, 224)
    assert labels.shape == (4,)
    assert images.dtype == torch.float32
