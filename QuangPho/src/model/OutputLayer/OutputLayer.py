import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextGatedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # 1. Gate nhìn thấy cả RGB và Spec (Input channels * 2)
        # 2. Dùng kernel_size=3 để hiểu ngữ cảnh không gian
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(channels // 2, channels, kernel_size=1, bias=False), # Nén lại về 1 channel gate
            nn.Sigmoid()
        )
        
        # Channel Attention để tinh chỉnh đặc trưng (SE Block - Optional nhưng rất tốt)
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.out_norm = nn.BatchNorm2d(channels)

    def forward(self, F_rgb, F_spec):
        """
        F_rgb, F_spec: (B, C, H, W)
        """
        # 1. Tính Gate dựa trên sự so sánh giữa cả 2 nguồn
        combined = torch.cat([F_rgb, F_spec], dim=1) # (B, 2C, H, W)
        spatial_gate = self.gate_net(combined)       # (B, C, H, W)
        
        # 2. Inject Spectral vào RGB có chọn lọc
        # Logic: F_rgb là gốc (spatial nét), F_spec là bổ trợ
        features = F_rgb + (F_spec * spatial_gate)
        
        # 3. Channel Attention (Tinh chỉnh lại lần cuối)
        # Giúp mạng tập trung vào các kênh quan trọng (ví dụ: kênh phản ánh màu xanh lá)
        channel_weight = self.se_block(features)
        features = features * channel_weight

        return self.out_norm(features)

class GlobalAvgMaxPool(nn.Module):
    def forward(self, x):
        # Giữ nguyên vì đã tối ưu
        avg = F.adaptive_avg_pool2d(x, 1)
        mx  = F.adaptive_max_pool2d(x, 1)
        return torch.cat([avg, mx], dim=1)   # (B, 2C, 1, 1)

class RobustWheatHead(nn.Module):
    def __init__(self, channels, num_classes=3, dropout_rate=0.3):
        super().__init__()
        
        # ✅ Global pooling: (B, C, H, W) → (B, C)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),  # (B, channels, 1, 1) → (B, channels)
            
            # Layer 1
            nn.Linear(channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            # Output
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Input: (B, channels, H, W)  ← Can be 7×7 or any size
        Output: (B, num_classes)
        """
        x = self.global_pool(x)  # (B, channels, 1, 1)
        return self.classifier(x)