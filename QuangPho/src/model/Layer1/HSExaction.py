import torch
import torch.nn as nn
from src.model.Layer1.HSSpetral import HSSpectralEncoder
from src.FeatureEngineering.HSFeature import HSFeature

class HSFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = HSSpectralEncoder()

    def forward(self, hs):
        f_sig = HSFeature.spectral_signature(hs)
        f_tex = HSFeature.spectral_texture(hs)
        f_dl  = self.encoder(hs)

        return torch.cat([f_sig, f_tex, f_dl], dim=1)
