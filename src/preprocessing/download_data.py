"""
Download Helper Script for Kaggle Dataset to Local / Colab
File: src/preprocessing/download_data.py
"""

import os
import shutil
import kagglehub


def download_and_extract(
    dataset_name: str = "aleksandarcvetanov/iq-othnccd-lung-cancer-augmented-dataset",
    destination: str = "data/raw/The IQ-OTHNCCD Lung Cancer Augmented Dataset"
) -> str:
    print(f"[*] Mengunduh dataset '{dataset_name}' via kagglehub...")
    download_path = kagglehub.dataset_download(dataset_name)
    print(f"[OK] Selesai diunduh ke cache: {download_path}")

    # Cari subfolder dataset
    subfolders = [os.path.join(download_path, d) for d in os.listdir(download_path) if os.path.isdir(os.path.join(download_path, d))]
    source_dir = subfolders[0] if subfolders else download_path

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if not os.path.exists(destination):
        print(f"[*] Menyalin dataset ke {destination}...")
        shutil.copytree(source_dir, destination)
        print(f"[OK] Dataset siap di: {destination}")
    else:
        print(f"[!] Folder tujuan sudah ada: {destination}")

    return destination


if __name__ == "__main__":
    download_and_extract()
