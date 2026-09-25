"""
Module: Data Splitting (Stratified Train/Val/Test)
File: src/preprocessing/split.py

Prinsip Codebase Design:
- Modul Deep: Menyembunyikan alur stratified K-Fold / train_test_split, verifikasi distribusi kelas,
  dan serialisasi artefak split ke disk.
- Interface sederhana: split_and_save(data_dir, output_dir, val_ratio, test_ratio, seed).
"""

import os
import json
from typing import Dict, List, Tuple
import yaml
from sklearn.model_selection import train_test_split
from src.preprocessing.dataset import scan_dataset


def split_dataset(
    samples: List[Tuple[str, int]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Membagi sampel citra secara stratified train/val/test berdasarkan label kelas.
    
    Args:
        samples: Pasangan (filepath, label_idx).
        val_ratio: Proporsi data validasi.
        test_ratio: Proporsi data pengujian (test set murni).
        seed: Random seed untuk reproduktibilitas.
        
    Returns:
        Dictionary dengan key 'train', 'val', 'test'.
    """
    labels = [s[1] for s in samples]
    
    n_total = len(samples)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    
    # 1. Pisahkan Train+Val dengan Test
    train_val_samples, test_samples, train_val_labels, _ = train_test_split(
        samples,
        labels,
        test_size=n_test,
        random_state=seed,
        stratify=labels
    )
    
    # 2. Pisahkan Train dengan Val
    train_samples, val_samples = train_test_split(
        train_val_samples,
        test_size=n_val,
        random_state=seed,
        stratify=train_val_labels
    )
    
    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples
    }


def split_and_save(
    data_dir: str,
    output_dir: str = "data/splits",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Dict[str, List[Tuple[str, int]]], Dict[str, int]]:
    """
    Interface utama: scan data mentah, hitung split stratified, dan simpan metadata split ke json.
    """
    os.makedirs(output_dir, exist_ok=True)
    samples, class_to_idx = scan_dataset(data_dir)
    
    splits = split_dataset(
        samples=samples,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )
    
    summary = {
        "class_to_idx": class_to_idx,
        "seed": seed,
        "total_samples": len(samples),
        "split_counts": {k: len(v) for k, v in splits.items()}
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
