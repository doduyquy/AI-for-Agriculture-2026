from src.model.Layer2.SoftCompetitionFusion import BiModalSoftCompetitionFusion
from src.model.Layer2.AttentionFusion import FusionAttention    
import torch
import torch.nn as nn
class FullFusionModel(nn.Module):
    def __init__(self,
                 dim_ms, dim_hs,
                 dim_ms_index, dim_ms_spec,
                 dim_sig, dim_tex,
                 embed_dim=128):
        super().__init__()

        # 1. Soft Competition (Gate trước)
        self.soft_comp = BiModalSoftCompetitionFusion(
            dim_ms=dim_ms,
            dim_hs=dim_hs,
            embed_dim=embed_dim,
            temperature=1.0
        )

        # 2. Cross Attention (Reasoning sau)
        self.fusion_attn = FusionAttention(
            dim_ms_index=dim_ms_index,
            dim_ms_spec=dim_ms_spec,
            dim_hs=embed_dim,      # dùng output đã fuse
            dim_sig=dim_sig,
            dim_tex=dim_tex,
            embed_dim=embed_dim,
            num_heads=4
        )

    def forward(self,
                F_ms, F_hs,
                F_ms_index, F_ms_spec,
                F_sig, F_tex):

        # ---- Stage 1: Soft Competition ----
        F_fused = self.soft_comp(F_ms, F_hs)
        # F_fused: (B, embed_dim)

        # ---- Stage 2: Cross Attention ----
        out = self.fusion_attn(
            F_ms_index=F_ms_index,
            F_ms_spec=F_ms_spec,
            F_hs=F_fused,     # HS đã được lọc
            F_sig=F_sig,
            F_tex_spec=F_tex
        )

        return out
