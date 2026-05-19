# Phúc lấy baseline là ResNet34 pretrained trên ImageNet nhé ^.^

import torch
import torch.nn as nn
from torchvision import models


class Resnet34(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, dropout=0.3):
        super().__init__()

        if pretrained:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
        else:
            weights = None

        # Dùng pretrained ResNet34 trên ImageNet
        self.backbone = models.resnet34(weights=weights)

        # ResNet34 layer4 output: [B, 512, 7, 7] nếu input [B, 3, 224, 224]
        in_features = self.backbone.fc.in_features  # 512

        # Thay classifier gốc bằng head phân loại 3 lớp
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def main():
    model = Resnet34(
        num_classes=3,
        pretrained=True,
        dropout=0.3,
    )

    x = torch.randn(4, 3, 224, 224)
    logits = model(x)

    print(logits.shape)  # torch.Size([4, 3])


if __name__ == "__main__":
    main()