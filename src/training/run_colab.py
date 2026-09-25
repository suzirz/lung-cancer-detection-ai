"""
Colab Auto-Runner Script (One-Command Training Pipeline)
File: src/training/run_colab.py

Skrip ini menjalankan seluruh alur otomatis tanpa intervensi manual:
1. Pengecekan hardware GPU Tesla T4.
2. Download dataset 12.184 citra (jika belum ada).
3. Split stratified dataset (train/val/test).
4. Eksekusi training baseline CNN (EfficientNet-B0) dengan FP16 mixed precision.
5. Evaluasi metrik dan pembuatan Confusion Matrix.
"""

import os
import sys
from pathlib import Path

# Pastikan root direktori proyek selalu berada di prioritas pertama sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PYTHONPATH"] = str(PROJECT_ROOT)

import subprocess
import torch
import yaml


def check_gpu() -> bool:
    print("\n" + "="*50)
    print("1. MEMERIKSA PERANGKAT KERAS (HARDWARE)")
    print("="*50)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[OK] GPU Terdeteksi: {gpu_name}")
        print(f"[OK] Total VRAM: {vram_gb:.2f} GB")
        return True
    else:
        print("[PERINGATAN] GPU CUDA tidak terdeteksi! Training akan berjalan di CPU (lambat).")
        print("Pastikan di Colab Anda telah memilih: Runtime -> Change runtime type -> T4 GPU.")
        return False


def run_pipeline():
    print("\n" + "="*50)
    print("MEMULAI TRAINING OTOMATIS LUNG CANCER DETECTION AI")
    print("="*50)

    # 1. Cek GPU
    check_gpu()

    # 2. Cek & Download Dataset
    dataset_dir = "data/raw/The IQ-OTHNCCD Lung Cancer Augmented Dataset"
    if not os.path.exists(dataset_dir):
        print("\n" + "="*50)
        print("2. MENGUNDUH DATASET (12.184 CITRA CT SCAN)")
        print("="*50)
        from src.preprocessing.download_data import download_and_extract
        download_and_extract()
    else:
        print(f"[OK] Dataset sudah tersedia di: {dataset_dir}")

    # 3. Split Data (Group-Aware — prevents augmentation leakage)
    print("\n" + "="*50)
    print("3. MEMBAGI DATASET (GROUP-AWARE STRATIFIED SPLIT)")
    print("="*50)
    splits_dir = "data/splits"
    splits_summary = os.path.join(splits_dir, "split_summary.json")

    # Check if existing splits use the old per-image method (leaky)
    force_resplit = False
    if os.path.exists(splits_summary):
        import json
        with open(splits_summary, "r", encoding="utf-8") as f:
            existing_summary = json.load(f)
        if existing_summary.get("split_method") != "group_aware_stratified":
            print("[!] Stale per-image splits detected — regenerating with group-aware method...")
            force_resplit = True
            # Remove old split artifacts
            import shutil
            shutil.rmtree(splits_dir, ignore_errors=True)

    if force_resplit or not os.path.exists(splits_summary):
        from src.preprocessing.split import split_and_save
        with open("configs/default.yaml", "r", encoding="utf-8") as f:
            cfg_split = yaml.safe_load(f)
        split_and_save(
            data_dir=cfg_split["data"]["raw_dir"],
            output_dir=cfg_split["data"]["splits_dir"],
            val_ratio=0.15,
            test_ratio=0.15,
            seed=cfg_split.get("seed", 42)
        )
    else:
        print(f"[OK] Group-aware splits already exist at: {splits_summary}")

    # 4. Eksekusi Training Model
    print("\n" + "="*50)
    print("4. MENJALANKAN TRAINING BASELINE CNN (EFFICIENTNET-B0 FP16)")
    print("="*50)
    from src.training.train import run_training
    run_training(config_path="configs/default.yaml")

    print("\n" + "="*50)
    print("[SUKSES] TRAINING OTOMATIS SELESAI LENGKAP!")
    print("="*50)
    print("File hasil:")
    print(" - Bobot Model Terbaik : models/baseline_efficientnet_b0_best.pth")
    print(" - Laporan Metrik     : reports/baseline_metrics.json")
    print(" - Confusion Matrix   : reports/baseline_confusion_matrix.png")


if __name__ == "__main__":
    run_pipeline()
