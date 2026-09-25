"""
Module: Preprocessing Dataset & DataLoaders
File: src/preprocessing/dataset.py

Prinsip Codebase Design:
- Modul Deep: Menyediakan interface sederhana (get_dataloaders) yang menyembunyikan
  kompleksitas transform, PyTorch Dataset, caching path, augmentasi medis, dan batching Tesla T4.
- Interface testable dan return result tanpa side-effects tak terduga.
"""

import os
import glob
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class LungCTDataset(Dataset):
    """
    PyTorch Dataset untuk citra CT scan paru (3 kelas: Benign, Malignant, Normal).
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[transforms.Compose] = None
    ) -> None:
        """
        Args:
            samples: List pasangan (file_path, class_idx).
            transform: Pipeline torchvision transforms.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        raw_path, label = self.samples[idx]
        img_path = raw_path.replace("\\", "/")
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_transforms(image_size: Tuple[int, int] = (224, 224)) -> Dict[str, transforms.Compose]:
    """
    Menghasilkan pipeline transform untuk training dan evaluasi.
    
    Training mencakup augmentasi data yang lazim pada citra CT scan (rotasi acak, horizontal/vertical flip)
    dan normalisasi sesuai standar pretrained ImageNet.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return {"train": train_transform, "eval": eval_transform}


def scan_dataset(data_dir: str) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
    """
    Memindai direktori dataset dan mengembalikan daftar sampel terlabel serta mapping kelas.
    """
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    samples: List[Tuple[str, int]] = []
    for cls_name in classes:
        cls_folder = os.path.join(data_dir, cls_name)
        file_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            file_paths.extend(glob.glob(os.path.join(cls_folder, "**", ext), recursive=True))
            
        for p in file_paths:
            portable_p = p.replace("\\", "/")
            samples.append((portable_p, class_to_idx[cls_name]))

    return samples, class_to_idx


def create_dataloaders(
    train_samples: List[Tuple[str, int]],
    val_samples: List[Tuple[str, int]],
    test_samples: List[Tuple[str, int]],
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[Dict[str, DataLoader], Dict[str, transforms.Compose]]:
    """
    Interface utama: menghasilkan DataLoader siap pakai untuk proses training dan pengujian.
    """
    tfms = get_transforms(image_size=image_size)

    train_ds = LungCTDataset(train_samples, transform=tfms["train"])
    val_ds = LungCTDataset(val_samples, transform=tfms["eval"])
    test_ds = LungCTDataset(test_samples, transform=tfms["eval"])

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
    }

    return loaders, tfms
