"""
Training Pipeline for Baseline CNN
File: src/training/train.py

Didesain untuk portabilitas tinggi:
- Otomatis memanfaatkan CUDA + FP16 Mixed Precision (NVIDIA Tesla T4 di Google Cloud)
- Fallback ke CPU secara transparan saat testing lokal
- Menyimpan bobot model terbaik berdasarkan Validation Recall/Loss ke models/
- Menyimpan log riwayat training ke reports/
"""

import os
import json
import time
import argparse
from typing import Dict, Any, Tuple, Optional
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from src.preprocessing.dataset import create_dataloaders
from src.models.cnn_extractor import build_model
from src.evaluation.metrics import calculate_metrics, plot_confusion_matrix


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool = True
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    val_loss = total_loss / len(all_labels)
    metrics = calculate_metrics(all_labels, all_preds, class_names)
    metrics["loss"] = val_loss
    return metrics, all_labels, all_preds


def run_training(config_path: str = "configs/default.yaml", max_epochs: Optional[int] = None) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Device Setup (Tesla T4 / CUDA / CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    use_amp = cfg.get("compute", {}).get("mixed_precision", True) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print("[*] FP16 Mixed Precision diaktifkan untuk NVIDIA Tensor Cores.")

    # 2. Load Dataset Splits
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

    # 3. DataLoaders
    batch_size = cfg["cnn"]["batch_size"]
    num_workers = cfg.get("compute", {}).get("num_workers", 2) if device.type == "cuda" else 0
    pin_memory = cfg.get("compute", {}).get("pin_memory", True) if device.type == "cuda" else False

    loaders, _ = create_dataloaders(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        batch_size=batch_size,
        image_size=tuple(cfg["data"]["image_size"]),
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # 4. Model Initialization
    backbone = cfg["cnn"]["backbone"]
    num_classes = cfg["data"]["num_classes"]
    model = build_model(backbone_name=backbone, num_classes=num_classes, pretrained=cfg["cnn"]["pretrained"])
    model = model.to(device)

    # 5. Optimizer & Criterion
    criterion = nn.CrossEntropyLoss()
    lr = float(cfg["cnn"]["learning_rate"])
    weight_decay = float(cfg["cnn"]["weight_decay"])
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = max_epochs if max_epochs is not None else cfg["cnn"]["epochs"]
    best_val_recall = -1.0
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, f"baseline_{backbone}_best.pth")

    print(f"[*] Memulai Training Baseline CNN ({backbone}) selama {epochs} epochs...")

    history = []
    for epoch in range(1, epochs + 1):
        start_t = time.time()
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp
        )

        val_metrics, _, _ = evaluate(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
            class_names=class_names
        )

        elapsed = time.time() - start_t
        val_recall = val_metrics["recall_macro"]
        val_acc = val_metrics["accuracy"]

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_acc*100:.2f}% Recall: {val_recall*100:.2f}%")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_acc,
            "val_recall_macro": val_recall
        })

        # Simpan checkpoint model terbaik (berdasarkan recall, prioritas medis)
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            torch.save(model.state_dict(), best_model_path)
            print(f" -> [Checkpoint Disimpan] Best Val Recall: {val_recall*100:.2f}% ke {best_model_path}")

    # 6. Evaluasi Final pada Test Set Murni
    print("\n[*] Menjalankan Evaluasi Final pada Test Set...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_metrics, y_true, y_pred = evaluate(
        model=model,
        loader=loaders["test"],
        criterion=criterion,
        device=device,
        class_names=class_names
    )

    print("="*50)
    print("HASIL EVALUASI BASELINE CNN (TEST SET)")
    print("="*50)
    print(f"Accuracy         : {test_metrics['accuracy']*100:.2f}%")
    print(f"Macro Recall     : {test_metrics['recall_macro']*100:.2f}% (Target PRD > 90%)")
    print(f"Weighted Recall  : {test_metrics['recall_weighted']*100:.2f}%")
    print(f"Macro Precision  : {test_metrics['precision_macro']*100:.2f}%")
    print(f"Macro F1-Score   : {test_metrics['f1_macro']*100:.2f}%")
    for name in class_names:
        print(f" - Recall {name:<15}: {test_metrics[f'recall_{name}']*100:.2f}%")
    print("="*50)

    # Simpan confusion matrix
    cm_path = "reports/baseline_confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, output_path=cm_path)
    print(f"[OK] Confusion matrix disimpan di: {cm_path}")

    # Simpan metrics JSON
    with open("reports/baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline CNN for Lung Cancer Detection")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    run_training(config_path=args.config, max_epochs=args.epochs)
