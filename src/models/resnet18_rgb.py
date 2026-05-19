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