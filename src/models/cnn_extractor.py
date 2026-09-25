"""
Module: CNN Feature Extractor & Baseline Classifier
File: src/models/cnn_extractor.py

Prinsip Codebase Design:
- Deep Module: Menyembunyikan instansiasi backbone pretrained (EfficientNet/ResNet),
  layer surgical replacement (memisahkan convolutional feature extraction dan classification head),
  serta mode forward (ekstraksi embedding vs logits klasifikasi).
- Interface sederhana: build_model(backbone, num_classes, pretrained) dan model.extract_features(x).
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


class LungCNNModel(nn.Module):
    """
    Model PyTorch CNN dengan kemampuan ganda:
    1. Klasifikasi End-to-End (Logits).
    2. Ekstraksi Fitur Vektor (Embeddings) untuk model Hybrid ML (Random Forest/XGBoost).
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        num_classes: int = 3,
        pretrained: bool = True
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes

        if "efficientnet" in self.backbone_name:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features
            
            # Pisahkan feature extractor dan classifier head
            self.features = base.features
            self.avgpool = base.avgpool
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features, num_classes)
            )
            self.embedding_dim = in_features

        elif "resnet" in self.backbone_name:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features
            
            self.features = nn.Sequential(*list(base.children())[:-2])
            self.avgpool = base.avgpool
            self.classifier = nn.Linear(in_features, num_classes)
            self.embedding_dim = in_features

        else:
            raise ValueError(f"Backbone tidak didukung: {backbone_name}. Pilih 'efficientnet_b0' atau 'resnet50'.")

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mengekstrak representasi vektor fitur berdimensi tinggi (embedding).
        Output bentuk: (batch_size, embedding_dim).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass end-to-end untuk klasifikasi baseline CNN (menghasilkan raw logits).
        """
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        return logits


def build_model(
    backbone_name: str = "efficientnet_b0",
    num_classes: int = 3,
    pretrained: bool = True
) -> LungCNNModel:
    """
    Factory function untuk menginisialisasi model.
    """
    return LungCNNModel(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained
    )
