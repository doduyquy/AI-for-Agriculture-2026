import torch
import torch.nn as nn
class HSSpectralEncoder(nn.Module):
    def __init__(self, in_bands=125, out_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_bands, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

    def forward(self, hs):
        """
        hs: (B, 125, H, W)
        """
        hs = hs.mean(dim=[2, 3])     # (B, 125)
        hs = hs.unsqueeze(-1)        # (B, 125, 1)

        f = self.encoder(hs)         # (B, out_dim, 1)
        return f.squeeze(-1)         # (B, out_dim)
