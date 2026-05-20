import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, dropout=0.3):
        super().__init__()
        
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Lấy in_features của classifier layer
        in_features = self.backbone.classifier[1].in_features
        
        # Thay classifier gốc bằng classifier mới phù hợp với num_classes
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        """Đóng băng toàn bộ backbone, chỉ train classifier."""
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("[Model] Backbone frozen – only classifier will be trained.")

    def unfreeze_layer4(self):
        """
        Mở đóng băng các block cuối của features (features.6, features.7, features.8) và classifier.
        Dùng tên 'layer4' cho tương thích với config của ResNet.
        """
        for name, param in self.backbone.named_parameters():
            if "classifier" in name or "features.6" in name or "features.7" in name or "features.8" in name:
                param.requires_grad = True
        print("[Model] Top feature blocks and classifier unfrozen.")

    def unfreeze_all(self):
        """Mở đóng băng toàn bộ tham số (fine-tune)."""
        for param in self.parameters():
            param.requires_grad = True
        print("[Model] All parameters unfrozen.")


def main():
    model = EfficientNetB0(
        num_classes=3,
        pretrained=True,
        dropout=0.3,
    )
    
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    
    print(logits.shape)  # Expected: torch.Size([4, 3])


if __name__ == "__main__":
    main()
