
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
import numpy as np

# --- GATED RESNET ARCHITECTURE (FIXED BACKBONE LOADING) ---
class GatedResNet(nn.Module):
    def __init__(self, num_classes, pretrained_path=None):
        super().__init__()
        # 125 bands
        ALL_BANDS = 125 
        self.gate = nn.Parameter(torch.zeros(ALL_BANDS))
        
        # Initialize Backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Adapt Conv1 for 125 channels
        old = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(ALL_BANDS, old.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        
        # --- FIX: Set FC before loading weights to match checkpoint shape ---
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        # Load Pretrained HS-Full Weights if available
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading backbone from {pretrained_path}")
            state_dict = torch.load(pretrained_path)
            try:
                self.backbone.load_state_dict(state_dict, strict=False)
            except Exception as e:
                # If loading fails, it might be due to keys mismatch, but we fixed the shape issue.
                print(f"Minor loading warning: {e}")
        else:
            print("Using Scaled ImageNet Weights for Conv1")
            with torch.no_grad():
                 self.backbone.conv1.weight[:] = old.weight.mean(1, keepdim=True).repeat(1, ALL_BANDS, 1, 1)

    def forward(self, x):
        g = torch.sigmoid(self.gate).view(1, -1, 1, 1)
        x = x * g
        return self.backbone(x)

# --- GATING MODEL TRAINING (With CE + L1 Loss - Approach A) ---
def train_gate_model(l1_lambda=0.005, epochs=20):
    print(f"\n--- Training Gating Model (L1 Lambda={l1_lambda}) ---")
    
    # Re-initialize DataLoaders to ensure fresh start
    t_ds = HSFlexibleDataset(HS_DIR, train_files, band_indices=None, augment=True, mean=full_mean, std=full_std)
    v_ds = HSFlexibleDataset(HS_DIR, val_files, band_indices=None, augment=False, mean=full_mean, std=full_std)
    t_loader = DataLoader(t_ds, batch_size=BATCH_SIZE, shuffle=True)
    v_loader = DataLoader(v_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    ckpt_path = r"D:\HocTap\NCKH_ThayDoNhuTai\Challenges\checkpoints\best_hs125_resnet18.pth"
    model = GatedResNet(num_classes=3, pretrained_path=ckpt_path).to(device)
    
    gate_params = [p for n, p in model.named_parameters() if 'gate' in n]
    backbone_params = [p for n, p in model.named_parameters() if 'gate' not in n]
    
    optimizer = optim.AdamW([
        {'params': gate_params, 'lr': 1e-2},
        {'params': backbone_params, 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    crit = nn.CrossEntropyLoss()
    
    for ep in range(epochs):
        model.train()
        
        gate_grad_sum = 0.0
        steps = 0
        loss_avg = 0.0
        
        for x, y in t_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            cel = crit(out, y)
            
            # --- APPROACH A: CE + L1 Sparsity ---
            # Penalize the mean of gate values to encourage sparsity
            # No Target-K constraint. We just want meaningful ranking.
            sig_gate = torch.sigmoid(model.gate)
            l1_loss = l1_lambda * sig_gate.mean()
            
            loss = cel + l1_loss
            
            loss.backward()
            
            if model.gate.grad is not None:
                gate_grad_sum += model.gate.grad.abs().mean().item()
                steps += 1
            optimizer.step()
            loss_avg += loss.item()
            
        # Validation
        model.eval()
        corr = 0; tot = 0
        with torch.no_grad():
            g_vals = torch.sigmoid(model.gate)
            g_std = g_vals.std().item()
            g_sum = g_vals.sum().item()
            g_min = g_vals.min().item()
            g_max = g_vals.max().item()
            
            for x, y in v_loader:
                out = model(x.to(device))
                corr += (out.argmax(1) == y.to(device)).sum().item()
                tot += x.size(0)
        acc = corr/tot
        
        print(f"Ep {ep+1}: Acc={acc:.3f} | Loss={loss_avg/len(t_loader):.3f}")
        print(f"   Gate: Sum={g_sum:.2f} | Std={g_std:.4f} | Min/Max={g_min:.3f}/{g_max:.3f}")
        
    return torch.sigmoid(model.gate).detach().cpu().numpy()

# Execute Gating Training
# gate_values = train_gate_model(l1_lambda=0.005, epochs=20)
# plt.plot(gate_values.flatten())
# plt.title("Band Importance (CE + L1)")
# plt.show()
