import torch
import torch.nn.functional as F
class HSFeature:
    @staticmethod
    def spectral_signature(hs):
        """
        hs: (B, C=125, H, W)
        return: (B, 125*3)
        """
        mean_spec = hs.mean(dim=[2, 3])          # (B, 125)
        std_spec  = hs.std(dim=[2, 3])           # (B, 125)

        # đạo hàm theo chiều phổ
        diff_spec = mean_spec[:, 1:] - mean_spec[:, :-1]
        diff_spec = F.pad(diff_spec, (0, 1))     # giữ size 125

        return torch.cat([mean_spec, std_spec, diff_spec], dim=1)
    @staticmethod
    def spectral_texture(hs):
        """
        hs: (B, 125, H, W)
        return: (B, 4)
        """
        spec = hs.mean(dim=[2, 3])   # (B, 125)

        mean = spec.mean(dim=1, keepdim=True)
        var  = spec.var(dim=1, keepdim=True)
        skew = ((spec - mean)**3).mean(dim=1, keepdim=True)
        kurt = ((spec - mean)**4).mean(dim=1, keepdim=True)

        return torch.cat([mean, var, skew, kurt], dim=1)

