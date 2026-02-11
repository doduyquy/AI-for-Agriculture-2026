import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision import transforms
import cv2
import numpy as np

class RGBCNNFeature(nn.Module):
    """
    Trích xuất đặc trưng không gian (shape + context) từ RGB
    Output: F_spatial_deep with spatial information preserved
    """
    def __init__(self, backbone='resnet18', pretrained=True):
        super().__init__()

        if backbone == 'resnet18':
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            self.feature_dim = base.fc.in_features  # 512

            # ✅ Giữ feature map, bỏ avgpool + fc
            self.encoder = nn.Sequential(
                *list(base.children())[:-2]  # [B, 512, 7, 7]
            )
            self.feature_map_size = 7  # ResNet18 output: 7×7

    def forward(self, x):
        """
        x: [B, 3, 224, 224]
        return: [B, 512, 7, 7]  ← ✅ Keep spatial features
        """
        return self.encoder(x)
    
    def forward_vector(self, x):
        """
        If global vector is needed (backward compatibility)
        x: [B, 3, H, W]
        return: [B, 512]
        """
        feat_map = self.encoder(x)  # [B, 512, 7, 7]
        return feat_map.mean(dim=[2, 3])  # [B, 512]
