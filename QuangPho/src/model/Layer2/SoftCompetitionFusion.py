import torch
import torch.nn as nn
import torch.nn.functional as F


class BiModalSoftCompetitionFusion(nn.Module):
    def __init__(self, dim_ms, dim_hs, embed_dim=128, temperature=1.0):
        super().__init__()

        self.temperature = temperature

        # Projection về không gian chung
        self.ms_proj = nn.Linear(dim_ms, embed_dim)
        self.hs_proj = nn.Linear(dim_hs, embed_dim)

        # Shared gating network (SE-style, nhẹ)
        self.attn_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 2 * embed_dim)  # 2 gates: MS & HS
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, F_ms, F_hs):
        # 1. Project
        q = self.ms_proj(F_ms)
        k = self.hs_proj(F_hs)

        # 2. Context & Gating
        context = torch.cat([q, k], dim=-1)
        gates = self.attn_net(context)
        gates = gates.view(-1, 2, q.shape[-1])
        
        # Soft Competition
        gates = F.softmax(gates / self.temperature, dim=1)
        gate_ms = gates[:, 0, :] 
        gate_hs = gates[:, 1, :] 

        # 3. Fusion Logic (Thay đổi quan trọng ở đây)
        
        # CÁCH 1: Weighted Average (Nội suy mượt mà) -> Khử nhiễu tốt nhất
        # Logic: Tại mỗi điểm, chỉ lấy thông tin từ nguồn tốt nhất.
        weighted_fused = (q * gate_ms) + (k * gate_hs)
        
        # CÁCH 2: Complementary Boosting (Tăng cường bổ sung)
        # Logic: Giữ lại cái chung (q+k)/2, cộng thêm phần nổi trội nhất.
        # fused = (q + k) * 0.5 + weighted_fused
        
        # Tôi khuyên dùng CÁCH 3: Residual Learning chuẩn (như ResNet)
        # Input gốc là (q+k). Mạng học cách sửa đổi (modulate) tổng này.
        base = q + k
        modulation = (q * gate_ms) + (k * gate_hs)
        
        # Output = Base + Modulation (được chuẩn hóa)
        # LayerNorm sẽ lo việc cân bằng biên độ
        return self.norm(base + modulation)