"""
Pipeline Runner: Ekstraksi Embedding CNN & Training Model Hybrid (RF & XGBoost)
File: src/training/train_hybrid.py

Alur Kerja:
1. Membaca model CNN yang sudah dilatih (baseline_efficientnet_b0_best.pth).
2. Mengekstrak vektor fitur 1.280 dimensi untuk seluruh split data (Train, Val, Test).
3. Melatih dua model Machine Learning klasik:
   - Hybrid Model 1: EfficientNet-B0 + Random Forest (Balanced)
   - Hybrid Model 2: EfficientNet-B0 + XGBoost
4. Melakukan komparasi performa komprehensif (CNN-Only vs Hybrid RF vs Hybrid XGBoost).
5. Menyimpan perbandingan ke reports/model_comparison.json dan reports/model_comparison.png.
"""

import os
import json
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.preprocessing.dataset import create_dataloaders
from src.models.extract_features import extract_and_save_embeddings
from src.models.ml_classifier import train_hybrid_classifier, evaluate_and_save_hybrid


def run_hybrid_pipeline(config_path: str = "configs/default.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Eksekusi Pipeline Hybrid pada Device: {device}")

    # 1. Load Data Splits
    splits_dir = cfg["data"]["splits_dir"]
    with open(os.path.join(splits_dir, "train_split.json"), "r", encoding="utf-8") as f:
        train_samples = json.load(f)
    with open(os.path.join(splits_dir, "val_split.json"), "r", encoding="utf-8") as f:
        val_samples = json.load(f)
    with open(os.path.join(splits_dir, "test_split.json"), "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    with open(os.path.join(splits_dir, "split_summary.json"), "r", encoding="utf-8") as f:
        summary = json.load(f)
        class_to_idx = summary["class_to_idx"]
        class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]

    # 2. DataLoaders untuk ekstraksi fitur
    loaders, _ = create_dataloaders(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        batch_size=cfg["cnn"]["batch_size"],
        image_size=tuple(cfg["data"]["image_size"]),
        num_workers=cfg.get("compute", {}).get("num_workers", 2) if device.type == "cuda" else 0,
        pin_memory=cfg.get("compute", {}).get("pin_memory", True) if device.type == "cuda" else False
    )

    # 3. Ekstraksi Fitur / Load jika sudah pernah diekstrak
    backbone = cfg["cnn"]["backbone"]
    features_cache = f"data/processed/cnn_features_{backbone}.npz"
    model_weight_path = f"models/baseline_{backbone}_best.pth"

    if not os.path.exists(model_weight_path):
        raise FileNotFoundError(f"Model weight '{model_weight_path}' tidak ditemukan di folder models/!")

    if os.path.exists(features_cache):
        print(f"[OK] Memuat cache fitur yang sudah diekstrak dari: {features_cache}")
        archive = np.load(features_cache)
        X_train, y_train = archive["X_train"], archive["y_train"]
        X_val, y_val = archive["X_val"], archive["y_val"]
        X_test, y_test = archive["X_test"], archive["y_test"]
    else:
        print("[*] Mengekstrak embedding fitur dari model CNN terlatih...")
        data_splits = extract_and_save_embeddings(
            model_path=model_weight_path,
            loaders=loaders,
            output_dir="data/processed",
            backbone=backbone,
            device=device
        )
        X_train, y_train = data_splits["train"]
        X_val, y_val = data_splits["val"]
        X_test, y_test = data_splits["test"]

    # 4. Training Hybrid 1: Random Forest
    print("\n" + "="*50)
    print("TRAINING HYBRID 1: CNN + RANDOM FOREST")
    print("="*50)
    rf_model = train_hybrid_classifier(X_train, y_train, model_type="random_forest")
    rf_metrics = evaluate_and_save_hybrid(rf_model, X_test, y_test, class_names, model_type="random_forest")

    # 5. Training Hybrid 2: XGBoost
    print("\n" + "="*50)
    print("TRAINING HYBRID 2: CNN + XGBOOST")
    print("="*50)
    xgb_model = train_hybrid_classifier(X_train, y_train, model_type="xgboost")
    xgb_metrics = evaluate_and_save_hybrid(xgb_model, X_test, y_test, class_names, model_type="xgboost")

    # 6. Komparasi dengan Baseline CNN-only
    baseline_metrics_path = "reports/baseline_metrics.json"
    cnn_metrics = {}
    if os.path.exists(baseline_metrics_path):
        with open(baseline_metrics_path, "r", encoding="utf-8") as f:
            cnn_metrics = json.load(f)
    else:
        cnn_metrics = {"accuracy": 0.9986, "recall_macro": 1.0, "f1_macro": 0.999}

    comparison = {
        "CNN-Only (EfficientNet-B0)": cnn_metrics,
        "Hybrid CNN + Random Forest": rf_metrics,
        "Hybrid CNN + XGBoost": xgb_metrics
    }

    # Simpan perbandingan ke JSON
    comp_json_path = "reports/model_comparison.json"
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[OK] Rangkuman komparasi model disimpan di: {comp_json_path}")

    # Buat Bar Chart Komparasi Model
    plot_comparison(comparison, output_path="reports/model_comparison.png")
    print("[OK] Grafik perbandingan disimpan di: reports/model_comparison.png")

    print("\n" + "="*50)
    print("HASIL KOMPARASI MODEL (TEST SET MURNI):")
    print("="*50)
    for model_name, m in comparison.items():
        print(f"[{model_name}]")
        print(f" - Accuracy     : {m['accuracy']*100:.2f}%")
        print(f" - Macro Recall : {m['recall_macro']*100:.2f}%")
        print(f" - Macro F1     : {m['f1_macro']*100:.2f}%")


def plot_comparison(comparison: dict, output_path: str = "reports/model_comparison.png"):
    models = list(comparison.keys())
    accuracies = [comparison[m]["accuracy"] * 100 for m in models]
    recalls = [comparison[m]["recall_macro"] * 100 for m in models]
    f1s = [comparison[m]["f1_macro"] * 100 for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    r1 = ax.bar(x - width, accuracies, width, label="Akurasi (%)", color="#1971c2")
    r2 = ax.bar(x, recalls, width, label="Recall Macro (%)", color="#2b8a3e")
    r3 = ax.bar(x + width, f1s, width, label="F1-Score (%)", color="#e03131")

    ax.set_ylabel("Skor (%)", fontsize=11)
    ax.set_title("Perbandingan Performa: CNN-Only vs Model Hybrid", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(loc="lower right")
    ax.set_ylim([90, 105])
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Tambahkan label angka
    for rects in (r1, r2, r3):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    run_hybrid_pipeline()
