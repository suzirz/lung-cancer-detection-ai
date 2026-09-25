"""
Module: Data Splitting (Group-Aware Stratified Train/Val/Test)
File: src/preprocessing/split.py

Prinsip Codebase Design:
- Modul Deep: Menyembunyikan alur group-aware stratified splitting, verifikasi distribusi kelas,
  dan serialisasi artefak split ke disk.
- Interface sederhana: split_and_save(data_dir, output_dir, val_ratio, test_ratio, seed).

CRITICAL FIX (Data Leakage Prevention):
The IQ-OTHNCCD augmented dataset contains multiple augmented variants per original scan
(e.g., 'Malignant case (445)(1).jpg' through '(445)(10).jpg'). Splitting per-image
causes augmented twins to appear in both train and test, inflating accuracy to 100%.

This module splits by PATIENT/SCAN GROUP — all augmented variants of the same scan
are assigned to the same split (train, val, OR test — never across splits).
"""

import os
import json
from typing import Dict, List, Tuple
from collections import defaultdict
import yaml
from sklearn.model_selection import train_test_split
from src.preprocessing.dataset import scan_dataset, extract_group_id


def split_dataset(
    samples: List[Tuple[str, int]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Group-aware stratified split: ensures all augmented variants of the same
    patient/scan stay in the same split partition.

    Algorithm:
    1. Extract group IDs from filenames (e.g., 'Malignant case_445')
    2. Build a mapping: group_id -> list of (filepath, label) samples
    3. Assign each group a single label (all samples in a group share the same class)
    4. Split GROUPS (not images) using stratified train_test_split
    5. Expand groups back to individual samples

    Args:
        samples: Pasangan (filepath, label_idx).
        val_ratio: Proporsi data validasi (applied to groups, not images).
        test_ratio: Proporsi data pengujian (applied to groups, not images).
        seed: Random seed untuk reproduktibilitas.

    Returns:
        Dictionary dengan key 'train', 'val', 'test', masing-masing berisi list sampel.
    """
    # 1. Group samples by patient/scan ID
    group_to_samples: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for filepath, label in samples:
        gid = extract_group_id(filepath)
        group_to_samples[gid].append((filepath, label))

    # 2. Build group-level arrays
    group_ids = list(group_to_samples.keys())
    # Each group has a single class label (all augmentations share the same label)
    group_labels = [group_to_samples[gid][0][1] for gid in group_ids]

    n_groups = len(group_ids)
    n_test_groups = max(1, int(round(n_groups * test_ratio)))
    n_val_groups = max(1, int(round(n_groups * val_ratio)))

    # 3. Split groups: first separate test, then split remainder into train/val
    train_val_gids, test_gids, train_val_labels, _ = train_test_split(
        group_ids,
        group_labels,
        test_size=n_test_groups,
        random_state=seed,
        stratify=group_labels
    )

    train_val_group_labels = [group_to_samples[gid][0][1] for gid in train_val_gids]

    train_gids, val_gids = train_test_split(
        train_val_gids,
        test_size=n_val_groups,
        random_state=seed,
        stratify=train_val_group_labels
    )

    # 4. Expand groups back to individual samples
    def expand_groups(gid_list):
        result = []
        for gid in gid_list:
            result.extend(group_to_samples[gid])
        return result

    splits = {
        "train": expand_groups(train_gids),
        "val": expand_groups(val_gids),
        "test": expand_groups(test_gids),
    }

    # 5. Verify zero group overlap (hard assertion — fail loud if violated)
    train_group_set = set(train_gids)
    val_group_set = set(val_gids)
    test_group_set = set(test_gids)

    assert len(train_group_set & val_group_set) == 0, "LEAKAGE: train/val group overlap"
    assert len(train_group_set & test_group_set) == 0, "LEAKAGE: train/test group overlap"
    assert len(val_group_set & test_group_set) == 0, "LEAKAGE: val/test group overlap"

    print(f"[OK] Group-aware split: {len(train_gids)} train groups, "
          f"{len(val_gids)} val groups, {len(test_gids)} test groups")
    print(f"[OK] Zero cross-split group overlap verified.")

    return splits


def split_and_save(
    data_dir: str,
    output_dir: str = "data/splits",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Dict[str, List[Tuple[str, int]]], Dict[str, int]]:
    """
    Interface utama: scan data mentah, hitung group-aware split stratified,
    dan simpan metadata split ke json.
    """
    os.makedirs(output_dir, exist_ok=True)
    samples, class_to_idx = scan_dataset(data_dir)

    splits = split_dataset(
        samples=samples,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    # Compute group stats for metadata
    group_counts = {}
    for split_name, split_data in splits.items():
        groups_in_split = set(extract_group_id(fp) for fp, _ in split_data)
        group_counts[split_name] = len(groups_in_split)

    summary = {
        "class_to_idx": class_to_idx,
        "seed": seed,
        "total_samples": len(samples),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "group_counts": group_counts,
        "split_method": "group_aware_stratified",
        "group_key": "patient_scan_id",
    }

    # Simpan metadata ringkasan
    summary_path = os.path.join(output_dir, "split_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Simpan daftar path & label per split
    for split_name, split_data in splits.items():
        split_path = os.path.join(output_dir, f"{split_name}_split.json")
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2)

    print(f"[OK] Split selesai: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    print(f"[OK] Disimpan di: {output_dir}")

    return splits, class_to_idx


if __name__ == "__main__":
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    split_and_save(
        data_dir=cfg["data"]["raw_dir"],
        output_dir=cfg["data"]["splits_dir"],
        val_ratio=0.15,
        test_ratio=0.15,
        seed=cfg.get("seed", 42)
    )
