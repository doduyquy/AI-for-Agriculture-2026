import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_spectral_indices(ms):
    """
    ms: Tensor (B, 5, H, W)
    return: Tensor (B, 3, H, W)
    """
    eps = 1e-6
    B, G, R, RE, NIR = ms[:,0], ms[:,1], ms[:,2], ms[:,3], ms[:,4]

    NDVI  = (NIR - R)  / (NIR + R  + eps)
    GNDVI = (NIR - G)  / (NIR + G  + eps)
    NDRE  = (NIR - RE) / (NIR + RE + eps)

    return torch.stack([NDVI, GNDVI, NDRE], dim=1)


class MSFeatureExtractor(nn.Module):
    @staticmethod
    def compute_spectral_indices(ms):
        """
        ms: Tensor (B, 5, H, W)
        return: Tensor (B, 3, H, W)
        """
        return compute_spectral_indices(ms)
