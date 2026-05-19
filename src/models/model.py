"""
model.py
Model builder theo kiến trúc kế thừa.

Hierarchy:
    BaseClassifier        (ABC)
    └── ResNetClassifier  – pretrained ResNet với custom FC
        └── ResNet18Classifier  – cụ thể cho ResNet-18
    MultimodalClassifier  – Late Fusion: RGB (ResNet18) + MS (CNN) + HS (CNN)
"""

from abc import ABC, abstractmethod
from typing import Optional, Any

import torch
import torch.nn as nn

from src.models import get_rgb_backbone_spec

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
        spec = get_rgb_backbone_spec(self.model_name)
        net = spec.model_fn(weights=spec.weights if self.pretrained else None)

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
    Tạo model ResNet (single-modality RGB) từ ConfigDict và đưa lên device.

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


# ──────────────────────────────────────────────
# MS Branch: CNN nhỏ cho 5 kênh đầu vào
# ──────────────────────────────────────────────
class MSBranch(nn.Module):
    """
    Lightweight CNN cho ảnh Multispectral (5 channels, 64x64).
    Output: feature vector 256 chiều.
    """

    def __init__(self, out_features: int = 256, dropout_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: 5 → 32 channels, 64x64 → 32x32
            nn.Conv2d(5, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32 → 64 channels, 32x32 → 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64 → 128 channels, 16x16 → 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 128 → 256 channels, 8x8 → 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),   # → (256, 1, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


# ──────────────────────────────────────────────
# HS Branch: CNN cho 125 kênh đầu vào (32x32)
# ──────────────────────────────────────────────
class HSBranch(nn.Module):
    """
    CNN cho ảnh Hyperspectral (125 channels, 32x32).
    Dùng Conv2d với in_channels=125; pooling aggressively vì spatial nhỏ.
    Output: feature vector 256 chiều.
    """

    def __init__(self, out_features: int = 256, dropout_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: 125 → 256 channels, 32x32 → 16x16
            # Dùng kernel 1x1 trước để giảm kênh (spectral mixing)
            nn.Conv2d(125, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 2: 256 → 256, spatial 32x32 → 16x16
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 256 → 256, 16x16 → 8x8
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Global average pool → (256, 1, 1)
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


# ──────────────────────────────────────────────
# MultimodalClassifier – Late Fusion
# ──────────────────────────────────────────────
class MultimodalClassifier(nn.Module):
    """
    Late Fusion model cho 3 modality: RGB, MS, HS.

    Architecture:
        RGB Branch  → ResNet18 (pretrained, FC removed) → 512-d
        MS Branch   → MSBranch (5 ch CNN)               → 256-d
        HS Branch   → HSBranch (125 ch CNN)              → 256-d
        Fusion Head → Concat(512+256+256=1024) → FC → num_classes

    Args:
        num_classes:     Số class phân loại.
        rgb_pretrained:  Sử dụng pretrained weights cho RGB branch.
        rgb_model:       Tên ResNet variant ('resnet18', 'resnet34', ...).
        ms_out:          Output dim của MS branch (default 256).
        hs_out:          Output dim của HS branch (default 256).
        dropout_p:       Dropout probability ở Fusion Head & branches.
        fusion_hidden:   Số neurons lớp ẩn trong Fusion Head (0 = bỏ qua).
    """

    def __init__(
        self,
        num_classes:    int,
        rgb_pretrained: bool  = True,
        rgb_model:      str   = "resnet18",
        ms_out:         int   = 256,
        hs_out:         int   = 256,
        dropout_p:      float = 0.3,
        fusion_hidden:  int   = 512,
    ):
        super().__init__()
        self.num_classes = num_classes

        # ── RGB Branch (ResNet backbone, bỏ FC cuối) ──────────────────────────
        spec = get_rgb_backbone_spec(rgb_model)
        rgb_feat = spec.out_features
        _resnet = spec.model_fn(weights=spec.weights if rgb_pretrained else None)
        # Bỏ lớp FC cuối → chỉ giữ feature extractor
        self.rgb_branch = nn.Sequential(*list(_resnet.children())[:-1])  # → (B, rgb_feat, 1, 1)
        self.rgb_flatten = nn.Flatten()

        # ── MS & HS Branches ─────────────────────────────────────────────────
        self.ms_branch = MSBranch(out_features=ms_out, dropout_p=dropout_p)
        self.hs_branch = HSBranch(out_features=hs_out, dropout_p=dropout_p)

        # ── Fusion Head ────────────────────────────────────────────────────────
        fused_dim = rgb_feat + ms_out + hs_out   # 512 + 256 + 256 = 1024
        if fusion_hidden > 0:
            self.fusion_head = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(fused_dim, fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
                nn.Linear(fusion_hidden, num_classes),
            )
        else:
            self.fusion_head = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(fused_dim, num_classes),
            )

    def forward(
        self,
        hs:  torch.Tensor,   # (B, 125, 32, 32)
        ms:  torch.Tensor,   # (B,   5, 64, 64)
        rgb: torch.Tensor,   # (B,   3, H,  W)
    ) -> torch.Tensor:       # (B, num_classes)
        rgb_feat = self.rgb_flatten(self.rgb_branch(rgb))  # (B, 512)
        ms_feat  = self.ms_branch(ms)                       # (B, 256)
        hs_feat  = self.hs_branch(hs)                       # (B, 256)

        fused = torch.cat([rgb_feat, ms_feat, hs_feat], dim=1)  # (B, 1024)
        return self.fusion_head(fused)                           # (B, num_classes)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def summary(self):
        total = self.count_parameters()
        rgb_p  = sum(p.numel() for p in self.rgb_branch.parameters())
        ms_p   = sum(p.numel() for p in self.ms_branch.parameters())
        hs_p   = sum(p.numel() for p in self.hs_branch.parameters())
        head_p = sum(p.numel() for p in self.fusion_head.parameters())
        print("[MultimodalClassifier]")
        print(f"  num_classes   : {self.num_classes}")
        print(f"  total params  : {total:,}")
        print(f"    rgb_branch  : {rgb_p:,}")
        print(f"    ms_branch   : {ms_p:,}")
        print(f"    hs_branch   : {hs_p:,}")
        print(f"    fusion_head : {head_p:,}")
        print(f"  device        : {next(self.parameters()).device}")

    def freeze_rgb_branch(self):
        """Đóng băng RGB branch để chỉ fine-tune MS, HS và Fusion Head."""
        for param in self.rgb_branch.parameters():
            param.requires_grad = False
        print("[MultimodalClassifier] RGB branch frozen.")

    def unfreeze_all(self):
        """Mở đóng băng toàn bộ tham số."""
        for param in self.parameters():
            param.requires_grad = True
        print("[MultimodalClassifier] All parameters unfrozen.")


def build_multimodal_model(
    cfg:    Any,
    device: torch.device,
) -> MultimodalClassifier:
    """
    Tạo MultimodalClassifier từ ConfigDict và đưa lên device.

    Các keys dùng trong cfg (có default an toàn):
        NUM_CLASSES       (bắt buộc)
        MODEL_NAME        → rgb_model (default 'resnet18')
        RGB_PRETRAINED    → rgb_pretrained (default True)
        MS_OUT_FEATURES   → ms_out (default 256)
        HS_OUT_FEATURES   → hs_out (default 256)
        DROPOUT_P         → dropout_p (default 0.3)
        FUSION_HIDDEN     → fusion_hidden (default 512)

    Returns:
        model đã .to(device)
    """
    model = MultimodalClassifier(
        num_classes    = cfg.NUM_CLASSES,
        rgb_pretrained = getattr(cfg, 'RGB_PRETRAINED',  True),
        rgb_model      = getattr(cfg, 'MODEL_NAME',      'resnet18'),
        ms_out         = getattr(cfg, 'MS_OUT_FEATURES', 256),
        hs_out         = getattr(cfg, 'HS_OUT_FEATURES', 256),
        dropout_p      = getattr(cfg, 'DROPOUT_P',       0.3),
        fusion_hidden  = getattr(cfg, 'FUSION_HIDDEN',   512),
    ).to(device)

    model.summary()
    return model
