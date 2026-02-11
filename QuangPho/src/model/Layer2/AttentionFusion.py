import torch
import torch.nn as nn

class FusionAttention(nn.Module):
    def __init__(self, 
                 dim_ms_index, dim_ms_spec,       # Input dims của nhóm MS
                 dim_hs, dim_sig, dim_tex,        # Input dims của nhóm HS
                 embed_dim=128,                   # Kích thước chung để tính Attention
                 num_heads=4,                     # Số lượng 'góc nhìn' (heads)
                 dropout=0.1):
        super().__init__()
        
        # 1. Projectors: Chiếu tất cả về cùng kích thước (embed_dim)
        # Để biến chúng thành các "Từ" (Tokens) có thể so sánh được
        self.proj_ms_idx  = nn.Linear(dim_ms_index, embed_dim)
        self.proj_ms_spec = nn.Linear(dim_ms_spec, embed_dim)
        
        self.proj_hs_raw  = nn.Linear(dim_hs, embed_dim)
        self.proj_hs_sig  = nn.Linear(dim_sig, embed_dim)
        self.proj_hs_tex  = nn.Linear(dim_tex, embed_dim)

        # 2. Cross Attention: Chuẩn Transformer
        # batch_first=True giúp input có dạng (Batch, Seq_Len, Dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # 3. Feed Forward & Norm (Cấu trúc chuẩn giúp hội tụ nhanh)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, F_ms_index, F_ms_spec, F_hs, F_sig, F_tex_spec):
        """
        Input là các vector đặc trưng (B, dim_x)
        Output là vector đã fuse (B, embed_dim)
        """
        B = F_ms_index.shape[0]

        # --- BƯỚC 1: TẠO TOKEN (TOKENIZATION) ---
        
        # Nhóm MS (Query): Tạo thành chuỗi có độ dài 2
        # unsqueeze(1) để biến (B, D) -> (B, 1, D)
        t_ms_idx  = self.proj_ms_idx(F_ms_index).unsqueeze(1)
        t_ms_spec = self.proj_ms_spec(F_ms_spec).unsqueeze(1)
        # Nối lại thành Query Sequence: (B, 2, embed_dim)
        query = torch.cat([t_ms_idx, t_ms_spec], dim=1) 

        # Nhóm HS (Key/Value): Tạo thành chuỗi có độ dài 3
        t_hs_raw = self.proj_hs_raw(F_hs).unsqueeze(1)
        t_hs_sig = self.proj_hs_sig(F_sig).unsqueeze(1)
        t_hs_tex = self.proj_hs_tex(F_tex_spec).unsqueeze(1)
        # Nối lại thành Key Sequence: (B, 3, embed_dim)
        key_value = torch.cat([t_hs_raw, t_hs_sig, t_hs_tex], dim=1)

        # --- BƯỚC 2: CROSS ATTENTION ---
        # Query (MS) hỏi Key (HS)
        # attn_output shape: (B, 2, embed_dim) - Giữ nguyên độ dài của Query
        attn_output, _ = self.cross_attn(query, key_value, key_value)

        # Residual connection + Norm
        query = self.norm1(query + attn_output)

        # --- BƯỚC 3: FEED FORWARD ---
        query = self.norm2(query + self.ffn(query))

        # --- BƯỚC 4: TỔNG HỢP ---
        # Hiện tại query có shape (B, 2, embed_dim).
        # Ta cần một vector duy nhất cho output. 
        # Có thể dùng Mean Pooling (trung bình cộng) hoặc lấy token đầu tiên.
        
        return query.mean(dim=1) # Trả về (B, embed_dim)
    #Dùng khi dữ liệu sạch và lớn


class MultiHeadSpectralAttention(nn.Module):
    """
    Multi-head attention for MS + HS fusion
    Used in Stage 1 pipeline with limited data
    
    Input: MS features (B, C_ms, H, W), HS features (B, C_hs, H, W)
    Output: Fused features (B, C_ms + C_hs, H, W)
    """
    def __init__(self, ms_channels, hs_channels, num_heads=4, dropout=0.1):
        super().__init__()
        self.ms_channels = ms_channels
        self.hs_channels = hs_channels
        self.num_heads = num_heads
        
        # Project MS and HS to common dimension for attention
        total_channels = ms_channels + hs_channels
        
        # Multi-head attention (spatial)
        self.attention = nn.MultiheadAttention(
            embed_dim=total_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(total_channels)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(total_channels, total_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_channels * 2, total_channels)
        )
        
    def forward(self, f_ms, f_hs):
        """
        Args:
            f_ms: MS features (B, C_ms, H, W)
            f_hs: HS features (B, C_hs, H, W)
        
        Returns:
            Fused features (B, C_ms + C_hs, H, W)
        """
        B, C_ms, H, W = f_ms.shape
        _, C_hs, _, _ = f_hs.shape
        
        # Concatenate MS and HS along channel dimension
        f_concat = torch.cat([f_ms, f_hs], dim=1)  # (B, C_ms + C_hs, H, W)
        
        # Reshape to sequence: (B, H*W, C_ms + C_hs)
        f_seq = f_concat.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        
        # Self-attention
        attn_out, _ = self.attention(f_seq, f_seq, f_seq)
        
        # Residual + Norm
        f_seq = self.norm(f_seq + attn_out)
        
        # Feed-forward + Residual
        f_seq = f_seq + self.ffn(f_seq)
        
        # Reshape back to spatial: (B, C, H, W)
        f_out = f_seq.permute(0, 2, 1).reshape(B, -1, H, W)
        
        return f_out