
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os

# Define the model class (copied from the notebook)
class HardConcreteGatedResNet18(nn.Module):
    # --- Hard-Concrete hyperparameters (cố định) ---
    BETA  = 2.0 / 3.0   # temperature
    GAMMA = -0.1         # stretch lower bound (< 0 để cho phép clamp = 0)
    ZETA  = 1.1          # stretch upper bound (> 1 để cho phép clamp = 1)
    EPS   = 1e-8         # tránh log(0)
    
    def __init__(self, num_bands=125, num_classes=3, log_alpha_init=2.0):
        super().__init__()
        
        # ====== Gate parameters ======
        self.log_alpha = nn.Parameter(torch.full((num_bands,), log_alpha_init))
        
        # ====== Backbone: ResNet18 pretrained ======
        base = models.resnet18(weights=None) # We only need structure
        
        # Sửa conv1 để nhận num_bands channels (thay vì 3 RGB)
        old_conv = base.conv1
        base.conv1 = nn.Conv2d(
            num_bands, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        
        # Sửa fc head
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.backbone = base
    
    def _hard_concrete_deterministic(self) -> torch.Tensor:
        try:
            # Check if log_alpha is defined
            if not hasattr(self, 'log_alpha'):
                return torch.zeros(125) # Dummy
            s = torch.sigmoid(self.log_alpha)
            s_bar = s * (self.ZETA - self.GAMMA) + self.GAMMA
            return torch.clamp(s_bar, min=0.0, max=1.0)
        except Exception:
            return torch.zeros(125) # Fail safe
    
    def gate(self) -> torch.Tensor:
        return self._hard_concrete_deterministic().detach()
    
    def get_topk_bands(self, k: int = 20):
        # We need to make sure the model is loaded properly
        if not hasattr(self, 'log_alpha'):
             # If log_alpha missing from state_dict, we can't do anything
             return [], []

        g = self.gate().cpu().numpy()
        sorted_idx = np.argsort(g)[::-1]
        topk_idx = sorted_idx[:k]
        topk_vals = g[topk_idx]
        return topk_idx, topk_vals

def main():
    checkpoint_path = r"D:\HocTap\NCKH_ThayDoNhuTai\Challenges\checkpoints\best_hs125_resnet18_hardconcrete.pth"
    
    if not os.path.exists(checkpoint_path):
        # Try relative path
        checkpoint_path = r"..\..\checkpoints\best_hs125_resnet18_hardconcrete.pth"
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize model structure
    # Note: HardConcreteGatedResNet18 structure must match the checkpoint.
    # The error might be in loading state_dict if keys don't match.
    # Let's inspect keys first if load fails.
    
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        print("Keys in state_dict:", list(state_dict.keys())[:5])
        
        # Check if 'log_alpha' is in keys
        if 'log_alpha' in state_dict:
            print("Found 'log_alpha' in checkpoint.")
            num_bands = state_dict['log_alpha'].shape[0]
            print(f"Number of bands in checkpoint: {num_bands}")
            
            # Extract log_alpha directly to avoid model instantiation issues
            log_alpha = state_dict['log_alpha']
            
            # Calculate gate values
            BETA  = 2.0 / 3.0
            GAMMA = -0.1
            ZETA  = 1.1
            
            s = torch.sigmoid(log_alpha)
            s_bar = s * (ZETA - GAMMA) + GAMMA
            gate_vals = torch.clamp(s_bar, min=0.0, max=1.0).cpu().numpy()
            
            sorted_idx = np.argsort(gate_vals)[::-1]
            topk_idx = sorted_idx[:20]
            topk_vals = gate_vals[topk_idx]
            
            print("\n=== Top 20 Bands (Extracted directly) ===")
            print("Indices:", list(topk_idx))
            print("Gate Values:", topk_vals)
            
        else:
            print("Error: 'log_alpha' not found in checkpoint state_dict.")
            
    except Exception as e:
        print(f"Failed to load/process checkpoint: {e}")
        return

if __name__ == "__main__":
    main()
