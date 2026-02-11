import torch
import torch.nn as nn
import torch.nn.functional as F
import src.FeatureEngineering.MSFeature as MSFeature

class MSFeatureExtractor(nn.Module):
    def __init__(self, num_bands=5, index_dim=3, out_dim=64, use_global_pool=True):
        super().__init__()
        self.use_global_pool = use_global_pool

        # 1. Spectral Encoder (Dùng Conv2d 1x1 để thay thế Conv1d pixel-wise)
        # Sử dụng 1x1 Conv hoạt động giống như MLP trên từng pixel nhưng giữ được shape (H,W)
        self.spec_mlp = nn.Sequential(
            nn.Conv2d(num_bands, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1), # Tăng độ sâu phi tuyến
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # 2. Spatial Context (Optional: Nếu muốn học kết cấu không gian)
        self.spatial_enc = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # 3. Projection cuối cùng
        self.final_conv = nn.Conv2d(32, out_dim, kernel_size=1)
        self.final_bn = nn.BatchNorm2d(out_dim)

        # 4. Index Embedding (Xử lý chỉ số phổ)
        self.index_mlp = nn.Sequential(
            nn.Linear(index_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32) # Output dimension khớp để concat nếu cần
        )
        
        # Cơ chế Attention cho việc gộp Features (Learnable weight)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, ms):
        """
        ms: (B, C, H, W)
        """
        B, C, H, W = ms.shape

        # --- Nhánh 1: Deep Learning trên Raw Bands ---
        # Không nhân weight vào ms gốc để tránh mất mát thông tin cho Conv
        f_spec = self.spec_mlp(ms)       # (B, 32, H, W)
        f_spec = self.spatial_enc(f_spec) # (B, 32, H, W) - Học không gian cục bộ
        f_spec = self.final_conv(f_spec)  # (B, out_dim, H, W)
        f_spec = F.relu(self.final_bn(f_spec))

        # --- Nhánh 2: Explicit Physics (Indices) ---
        # Tính trên ảnh gốc để bảo toàn ý nghĩa vật lý
        with torch.no_grad(): # Thường indices là công thức cố định, không cần grad ngược qua ms
            indices = MSFeature.compute_spectral_indices(ms) # (B, index_dim, H, W)
        
        # Nếu muốn ghép indices vào features:
        # Cần xử lý indices giống dạng feature map hoặc global vector
        if self.use_global_pool:
            # Pooling cho bài toán Classification
            v_spec = F.adaptive_avg_pool2d(f_spec, 1).flatten(1) # (B, out_dim)
            
            v_ind = indices.mean(dim=[2, 3]) # (B, index_dim)
            v_ind = self.index_mlp(v_ind)    # (B, 32)
            
            return torch.cat([v_spec, v_ind], dim=1) # (B, out_dim + 32)
        
        else:
            # Giữ nguyên không gian cho Segmentation (B, C_out, H, W)
            # Cần chiếu indices sang cùng chiều không gian feature để cộng/cat
            # Ở đây ta trả về feature map từ nhánh Deep Learning là chính
            return f_spec