# Phúc lấy baseline là resnet18 pretain trên Imagenet nhé ^.^
import torch
import torch.nn as nn
from torchvision import models
class Resnet18(nn.Module):
    def __init__(self,num_classes=3,pretrained=True,dropout=0.3):
        super().__init__()
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights=None
        self.backbone=models.resnet18(weights=weights) # dùng pretrain cảu resnt18 trên Imagenet
        in_features=self.backbone.fc.in_features # layer4: [B, 512, 7, 7]
        self.backbone.fc=nn.Sequential( # qua MLP
            nn.Dropout(dropout),
            nn.Linear(in_features,num_classes)
        )
    def forward(self,x):
        return self.backbone(x) 

    def freeze_backbone(self):
        """Đóng băng toàn bộ backbone, chỉ train FC head."""
        for name, param in self.backbone.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        print("[Model] Backbone frozen – only FC head will be trained.")

    def unfreeze_layer4(self):
        """Mở đóng băng layer4 và FC, các layer trước vẫn đóng băng."""
        for name, param in self.backbone.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
        print("[Model] Layer4 and FC unfrozen.")

    def unfreeze_all(self):
        """Mở đóng băng toàn bộ tham số (fine-tune)."""
        for param in self.parameters():
            param.requires_grad = True
        print("[Model] All parameters unfrozen.")
    
def main():
    model = Resnet18(
    num_classes=3,
    pretrained=True,
    dropout=0.3,
)

    x = torch.randn(4, 3, 224, 224)
    logits = model(x)

    print(logits.shape)
if __name__ == "__main__":
    main()