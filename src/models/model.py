import torch.nn as nn
import torchvision.models as models

class BaseModel(nn.Module):
    """Base class for all models."""
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented")

class ResNet18Model(BaseModel):
    """ResNet18 model for RGB image classification."""
    def __init__(self, num_classes=3, pretrained=True):
        super(ResNet18Model, self).__init__()
        
        # Load pretrained ResNet18
        if pretrained:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.model = models.resnet18(weights=None)
            
        # Replace the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
