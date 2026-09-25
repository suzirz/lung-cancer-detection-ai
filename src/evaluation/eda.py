"""
Exploratory Data Analysis (EDA) Script for Lung Cancer Detection AI.

Menganalisis dataset citra CT scan paru (distribusi kelas, resolusi,
statistik pixel, integritas file) dan menyimpan visualisasi ringkasan.
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Membaca konfigurasi proyek."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def inspect_dataset(data_dir: str) -> Tuple[Dict[str, int], List[dict]]:
    """
    Menghitung jumlah gambar per kelas dan mengumpulkan informasi metadata citra.
    """
    class_counts = {}
    sample_records = []
    
    # Deteksi kelas dari subfolder
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for cls in subdirs:
        cls_path = os.path.join(data_dir, cls)
        # Ambil semua file citra di subfolder (termasuk Train/Test jika ada)
        images = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            images.extend(glob.glob(os.path.join(cls_path, "**", ext), recursive=True))
            
        class_counts[cls] = len(images)
        
        # Ambil sampel metadata dari 10 citra per kelas
        for img_path in images[:10]:
            try:
                with Image.open(img_path) as img:
                    arr = np.array(img)
                    sample_records.append({
                        "class": cls,
                        "path": img_path,
                        "size": img.size, # (width, height)
                        "mode": img.mode,
                        "mean_intensity": float(np.mean(arr)),
                        "std_intensity": float(np.std(arr)),
                        "min_val": int(np.min(arr)),
                        "max_val": int(np.max(arr))
                    })
            except Exception as e:
                print(f"[WARNING] Gagal membaca gambar {img_path}: {e}")
                
    return class_counts, sample_records


def generate_eda_report(data_dir: str, output_dir: str = "reports/eda") -> None:
    """Menjalankan analisis lengkap dan menghasilkan plot grafik serta ringkasan teks."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Melakukan scan dataset di: {data_dir}")
    
    counts, samples = inspect_dataset(data_dir)
    total_images = sum(counts.values())
    
    print("\n" + "="*50)
    print("HASIL EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*50)
    print(f"Total Citra Terdeteksi : {total_images:,}")
    for cls_name, count in counts.items():
        pct = (count / total_images * 100) if total_images > 0 else 0
        print(f" - {cls_name:<20}: {count:>6,} ({pct:.2f}%)")
    print("="*50)

    # 1. Bar Chart Distribusi Kelas
    plt.figure(figsize=(8, 5))
    classes = list(counts.keys())
    values = list(counts.values())
    colors = ["#2b8a3e", "#e03131", "#1971c2"][:len(classes)]
    
    bars = plt.bar(classes, values, color=colors, edgecolor="black", alpha=0.85)
    plt.title("Distribusi Kelas Dataset CT Scan Paru", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Kategori Diagnosa", fontsize=12)
    plt.ylabel("Jumlah Citra", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + (max(values) * 0.015),
                 f"{int(h):,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
                 
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[OK] Grafik distribusi disimpan ke: {chart_path}")

    # 2. Visualisasi Sampel Citra
    fig, axes = plt.subplots(len(classes), 3, figsize=(10, 3 * len(classes)))
    if len(classes) == 1:
        axes = np.expand_dims(axes, 0)
        
    for i, cls in enumerate(classes):
        cls_samples = [s for s in samples if s["class"] == cls][:3]
        for j, s in enumerate(cls_samples):
            img = Image.open(s["path"])
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_title(f"{cls}\n{s['size'][0]}x{s['size'][1]}", fontsize=10)
            ax.axis("off")
            
    plt.tight_layout()
    samples_path = os.path.join(output_dir, "sample_images.png")
    plt.savefig(samples_path, dpi=300)
    plt.close()
    print(f"[OK] Visualisasi sampel citra disimpan ke: {samples_path}")


if __name__ == "__main__":
    cfg = load_config("configs/default.yaml")
    target_data_dir = cfg["data"]["raw_dir"]
    generate_eda_report(target_data_dir)
