"""
model.py
Model builder theo kiến trúc kế thừa.

Hierarchy:
    BaseClassifier        (ABC)
    └── ResNetClassifier  – pretrained ResNet với custom FC
        └── ResNet18Classifier  – cụ thể cho ResNet-18
"""

from abc import ABC, abstractmethod
from typing import Optional, Any

import torch
import torch.nn as nn
import torchvision.models as models

# ──────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────
class BaseClassifier(ABC, nn.Module):
    """
    Base class cho mọi classifier trong project.
    Subclass bắt buộc implement: build_backbone()
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.backbone    = self.build_backbone()

    @abstractmethod
    def build_backbone(self) -> nn.Module:
        """Trả về backbone đã gắn FC head phù hợp."""
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def summary(self):
        print(f"[{self.__class__.__name__}]")
        print(f"  num_classes : {self.num_classes}")
        print(f"  parameters  : {self.count_parameters():,}")
        print(f"  device      : {next(self.parameters()).device}")


# ──────────────────────────────────────────────
# Generic ResNet classifier
# ──────────────────────────────────────────────
class ResNetClassifier(BaseClassifier):
    """
    Wrapper cho bất kỳ variant ResNet nào.
    Thay thế FC cuối bằng Linear(in_features, num_classes).
    """

    SUPPORTED = {
        "resnet18" : (models.resnet18,  models.ResNet18_Weights.IMAGENET1K_V1),
        "resnet34" : (models.resnet34,  models.ResNet34_Weights.IMAGENET1K_V1),
        "resnet50" : (models.resnet50,  models.ResNet50_Weights.IMAGENET1K_V1),
        "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
    }

    def __init__(
        self,
        num_classes: int,
        model_name:  str  = "resnet18",
        pretrained:  bool = True,
        dropout_p:   float = 0.0,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.dropout_p  = dropout_p
        super().__init__(num_classes)   # gọi build_backbone() bên trong

    def build_backbone(self) -> nn.Module:
        if self.model_name not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported model: {self.model_name}. "
                f"Choose from: {list(self.SUPPORTED.keys())}"
            )

        model_fn, weights = self.SUPPORTED[self.model_name]
        net = model_fn(weights=weights if self.pretrained else None)

        in_features = net.fc.in_features
        if self.dropout_p > 0.0:
            net.fc = nn.Sequential(
                nn.Dropout(p=self.dropout_p),
                nn.Linear(in_features, self.num_classes),
            )
        else:
            net.fc = nn.Linear(in_features, self.num_classes)

        return net

    def freeze_backbone(self):
        """Đóng băng toàn bộ backbone, chỉ train FC head."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        print("[Model] Backbone frozen – only FC head will be trained.")

    def unfreeze_all(self):
        """Mở đóng băng toàn bộ tham số (fine-tune)."""
        for param in self.parameters():
            param.requires_grad = True
        print("[Model] All parameters unfrozen.")


# ──────────────────────────────────────────────
# Concrete: ResNet-18 (dùng trong baseline)
# ──────────────────────────────────────────────
class ResNet18Classifier(ResNetClassifier):
    """
    ResNet-18 cụ thể – shortcut tiện dùng.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained:  bool  = True,
        dropout_p:   float = 0.0,
    ):
        super().__init__(
            num_classes=num_classes,
            model_name="resnet18",
            pretrained=pretrained,
            dropout_p=dropout_p,
        )


# ──────────────────────────────────────────────
# Factory function (tiện dùng từ config)
# ──────────────────────────────────────────────
def build_model(
    cfg:        Any,
    device:     torch.device,
    pretrained: bool  = True,
    dropout_p:  float = 0.0,
) -> ResNetClassifier:
    """
    Tạo model từ ConfigDict và đưa lên device.

    Returns:
        model đã .to(device)
    """
    model = ResNetClassifier(
        num_classes=cfg.NUM_CLASSES,
        model_name=cfg.MODEL_NAME,
        pretrained=pretrained,
        dropout_p=dropout_p,
    ).to(device)

    model.summary()
    return model
