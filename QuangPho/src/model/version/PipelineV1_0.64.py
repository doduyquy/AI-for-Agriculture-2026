"""
PipelineV1_Stage1_TwoStep.py - TWO-STEP BINARY CLASSIFICATION

🎯 Mục tiêu Stage 1:
  - Step 1: Phân biệt Healthy vs Diseased (RGB-dominant)
  - Step 2: Phân biệt Rust vs Other (Spectral-dominant, chỉ khi diseased)

✅ Ưu điểm:
  - Giải quyết vấn đề Health bị nhầm với Rust
  - Phù hợp với quy trình chẩn đoán thực tế
  - Tận dụng điểm mạnh của từng modality
  - Tránh confusion giữa Health và Rust

🔧 Cấu hình:
  - Phase 1: Train Step 1 (20 epochs, binary: Healthy vs Diseased)
  - Phase 2: Train Step 2 (20 epochs, binary: Rust vs Other, on diseased only)
  - Phase 3: Fine-tune end-to-end (10 epochs, 3-class)
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import subprocess
import cv2

# Install dependencies
try:
    import tifffile
except ImportError:
    print("Installing tifffile...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tifffile"])
    import tifffile

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("\n" + "="*70)
    print("ERROR: albumentations is not installed!")
    print("Please install it manually using:")
    print("  pip install albumentations")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader, Subset

from src.model.Layer1.CNNRGB import RGBCNNFeature
from src.model.Layer2.AttentionFusion import MultiHeadSpectralAttention
from src.FeatureEngineering.RGBFeature import RGBFeature


# ==================== ASYMMETRIC BIASED CONTEXT GATE ====================
class AsymmetricBiasedGate(nn.Module):
    """
    Biased gate handling ASYMMETRIC dimensions
    RGB: 512 channels (from frozen ResNet18)
    Spectral: 256 channels (from trainable encoders)
    
    Args:
        rgb_channels: Number of RGB channels (default: 512)
        spec_channels: Number of Spectral channels (default: 256)
        rgb_weight: Weight for RGB (0-1), spectral_weight = 1 - rgb_weight
    """
    def __init__(self, rgb_channels=512, spec_channels=256, rgb_weight=0.65):
        super().__init__()
        self.rgb_channels = rgb_channels
        self.spec_channels = spec_channels
        self.rgb_weight = rgb_weight
        self.spec_weight = 1.0 - rgb_weight
        
        # Gate network: handles rgb_channels + spec_channels
        total_channels = rgb_channels + spec_channels  # 512 + 256 = 768
        hidden_channels = (rgb_channels + spec_channels) // 2  # 384
        
        self.gate = nn.Sequential(
            nn.Conv2d(total_channels, hidden_channels, 1),  # 768→384
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, 1),  # 384→1
            nn.Sigmoid()
        )
        
        # Project spectral to match RGB dimension for fusion
        if spec_channels != rgb_channels:
            self.spec_proj = nn.Conv2d(spec_channels, rgb_channels, 1)  # 256→512
        else:
            self.spec_proj = nn.Identity()
    
    def forward(self, f_rgb, f_spec):
        """
        Args:
            f_rgb: RGB features (B, 512, H, W)
            f_spec: Spectral features (B, 256, H, W)
        
        Returns:
            Fused features (B, 512, H, W)
        """
        # Compute gate using both features
        concat = torch.cat([f_rgb, f_spec], dim=1)  # (B, 768, H, W)
        gate = self.gate(concat)  # (B, 1, H, W)
        
        # Project spectral to match RGB dimension
        f_spec_proj = self.spec_proj(f_spec)  # (B, 256, H, W) → (B, 512, H, W)
        
        # Biased fusion (output is 512 channels)
        fused = self.rgb_weight * f_rgb + self.spec_weight * gate * f_spec_proj
        
        return fused  # (B, 512, H, W)


# ==================== FOCAL LOSS ====================
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in Step 1
    
    Args:
        alpha: Weighting factor for class imbalance (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
               - gamma=0: equivalent to CrossEntropyLoss
               - gamma>0: down-weight easy examples, focus on hard ones
        reduction: 'mean', 'sum', or 'none'
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C) - model predictions (logits)
            targets: (B,) - ground truth labels
        
        Returns:
            Focal loss value
        """
        # Compute standard cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute p_t (probability of true class)
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss: FL = alpha * (1 - pt)^gamma * CE
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ==================== TWO-STEP WHEAT NET ====================
class TwoStepWheatNet(nn.Module):
    """
    Two-Step Binary Classification for Wheat Disease Detection
    
    Step 1: Healthy vs Diseased (RGB-dominant, 85% RGB, 15% Spectral)
            - Focus on visual appearance (green vs not green)
            - Chlorophyll content difference
    
    Step 2: Rust vs Other (Spectral-dominant, 30% RGB, 70% Spectral)
            - Focus on disease-specific spectral signatures
            - Iron oxide bands for Rust detection
            - Only applied to diseased samples
    
    Architecture:
        Shared: ResNet18 (frozen), MS encoder, HS encoder, Spectral fusion
        Step 1: Biased gate (85% RGB) + Binary head
        Step 2: Biased gate (30% RGB) + Binary head
    """
    def __init__(self, dropout_rate=0.4):
        super().__init__()
        
        print("\n🏗️ Building TwoStepWheatNet...")
        
        # ===== SHARED FEATURE EXTRACTORS =====
        print("  1. RGB Branch (ResNet18, FROZEN)...")
        self.rgb_backbone = RGBCNNFeature(backbone='resnet18', pretrained=True)
        for param in self.rgb_backbone.parameters():
            param.requires_grad = False
        
        print("  2. MS Encoder (5→128 channels, TRAINABLE)...")
        self.ms_encoder = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        print("  3. HS Encoder (125→128 channels, TRAINABLE)...")
        self.hs_encoder = nn.Sequential(
            nn.Conv2d(125, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        print("  4. Spectral Attention Fusion (MS+HS, TRAINABLE)...")
        self.spectral_attention = MultiHeadSpectralAttention(
            ms_channels=128,
            hs_channels=128,
            num_heads=4,
            dropout=dropout_rate
        )
        
        # ⭐ NEW: Keep 256 channels (no expansion to 512)
        print("  4a. Spectral Projection (KEEP 256 channels - no expansion!)...")
        self.spectral_proj = nn.Sequential(
            nn.BatchNorm2d(256),  # Keep 256 channels!
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        
        # ===== RGB HANDCRAFTED FEATURES (for Step 1 only) =====
        print("  4b. RGB Handcrafted Feature Projection (84→128, TRAINABLE)...")
        # GLCM (36) + LBP multiscale (30) + Color (18) = 84 dims
        self.rgb_handcrafted_proj = nn.Sequential(
            nn.Linear(84, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128)
        )
        
        # Fusion layer for CNN + Handcrafted
        self.rgb_fusion = nn.Linear(512 + 128, 512)
        
        # ===== STEP 1: HEALTHY vs DISEASED (RGB-DOMINANT) =====
        print("  5. Step 1 Components (Healthy vs Diseased, 65% RGB + Handcrafted)...")
        self.step1_gate = AsymmetricBiasedGate(rgb_channels=512, spec_channels=256, rgb_weight=0.65)
        self.step1_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # Binary: Healthy (0) vs Diseased (1)
        )
        
        # ===== STEP 2: RUST vs OTHER (SPECTRAL-DOMINANT) =====
        print("  6. Step 2 Components (Rust vs Other, 70% Spectral)...")
        self.step2_gate = AsymmetricBiasedGate(rgb_channels=512, spec_channels=256, rgb_weight=0.30)
        self.step2_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # Binary: Rust (0) vs Other (1)
        )
        
        print("✅ Model built successfully!\n")
    
    def forward(self, rgb, ms, hs, rgb_handcrafted=None, step='both'):
        """
        Args:
            rgb, ms, hs: Input tensors
            rgb_handcrafted: RGB handcrafted features (B, 75) - optional
            step: 'step1', 'step2', or 'both'
        
        Returns:
            If step='step1': (B, 2) - Healthy vs Diseased logits
            If step='step2': (B, 2) - Rust vs Other logits
            If step='both': (B, 3) - Final 3-class logits
        """
        # Extract shared features
        with torch.no_grad():
            f_rgb_cnn = self.rgb_backbone(rgb)  # (B, 512, 7, 7)
        
        f_ms = self.ms_encoder(ms)  # (B, 128, 7, 7)
        f_hs = self.hs_encoder(hs)  # (B, 128, 7, 7)
        f_spec = self.spectral_attention(f_ms, f_hs)  # (B, 256, 7, 7)
        f_spec = self.spectral_proj(f_spec)  # (B, 256, 7, 7) ⭐ Keep 256!
        
        # Enhance RGB features with handcrafted features (for Step 1)
        if rgb_handcrafted is not None and step in ['step1', 'both']:
            # Pool CNN features
            f_rgb_pooled = F.adaptive_avg_pool2d(f_rgb_cnn, (1, 1)).flatten(1)  # (B, 512)
            # Project handcrafted features
            f_handcrafted = self.rgb_handcrafted_proj(rgb_handcrafted)  # (B, 128)
            # Fuse
            f_rgb_fused = torch.cat([f_rgb_pooled, f_handcrafted], dim=1)  # (B, 640)
            f_rgb_fused = self.rgb_fusion(f_rgb_fused)  # (B, 512)
            # Reshape back to spatial
            f_rgb = f_rgb_fused.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)  # (B, 512, 7, 7)
        else:
            f_rgb = f_rgb_cnn
        
        if step == 'step1':
            # Only Step 1
            f_step1 = self.step1_gate(f_rgb, f_spec)
            return self.step1_head(f_step1)
        
        elif step == 'step2':
            # Only Step 2
            f_step2 = self.step2_gate(f_rgb, f_spec)
            return self.step2_head(f_step2)
        
        else:  # step == 'both'
            # Both steps for final 3-class prediction
            f_step1 = self.step1_gate(f_rgb, f_spec)
            step1_logits = self.step1_head(f_step1)  # (B, 2)
            
            f_step2 = self.step2_gate(f_rgb, f_spec)
            step2_logits = self.step2_head(f_step2)  # (B, 2)
            
            # Convert to 3-class logits
            step1_probs = F.softmax(step1_logits, dim=1)
            step2_probs = F.softmax(step2_logits, dim=1)
            
            # P(Healthy) = P(Healthy | Step1)
            # P(Rust) = P(Diseased | Step1) * P(Rust | Step2)
            # P(Other) = P(Diseased | Step1) * P(Other | Step2)
            p_healthy = step1_probs[:, 0].unsqueeze(1)
            p_rust = (step1_probs[:, 1] * step2_probs[:, 0]).unsqueeze(1)
            p_other = (step1_probs[:, 1] * step2_probs[:, 1]).unsqueeze(1)
            
            final_probs = torch.cat([p_healthy, p_rust, p_other], dim=1)
            final_logits = torch.log(final_probs + 1e-8)
            
            return final_logits


# ==================== DATA LOADING ====================
def load_data_from_folder(data_root, split='train'):
    """Load RGB, MS, HS paths and labels from folder"""
    rgb_folder = data_root / split / 'RGB'
    ms_folder = data_root / split / 'MS'
    hs_folder = data_root / split / 'HS'
    
    if not rgb_folder.exists():
        return [], [], [], []
    
    rgb_paths, ms_paths, hs_paths, labels = [], [], [], []
    
    # Get all RGB files
    rgb_files = sorted(list(rgb_folder.glob('*.png')) + list(rgb_folder.glob('*.tif')))
    
    for rgb_file in rgb_files:
        filename = rgb_file.name
        stem = rgb_file.stem
        
        # Determine class from filename
        if filename.startswith('Health'):
            label = 0
        elif filename.startswith('Rust'):
            label = 1
        elif filename.startswith('Other'):
            label = 2
        else:
            continue
        
        # Find corresponding MS and HS files
        ms_file = ms_folder / f"{stem}.tif"
        hs_file = hs_folder / f"{stem}.tif"
        
        if ms_file.exists() and hs_file.exists():
            rgb_paths.append(str(rgb_file))
            ms_paths.append(str(ms_file))
            hs_paths.append(str(hs_file))
            labels.append(label)
    
    return rgb_paths, ms_paths, hs_paths, labels


# ==================== DATASET ====================
class Stage1Dataset(torch.utils.data.Dataset):
    """Dataset with light augmentation for Stage 1 + RGB handcrafted features"""
    def __init__(self, rgb_paths, ms_paths, hs_paths, labels, augment=False, use_handcrafted=True):
        self.rgb_paths = rgb_paths
        self.ms_paths = ms_paths
        self.hs_paths = hs_paths
        self.labels = labels
        self.augment = augment
        self.use_handcrafted = use_handcrafted
        
        # RGB augmentation
        if augment:
            self.rgb_transform = A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
                A.CenterCrop(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.rgb_transform = A.Compose([
                A.Resize(256, 256),
                A.CenterCrop(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        # MS/HS transforms (same for both, applied separately)
        self.get_ms_transform = lambda: A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5) if augment else A.NoOp(),
            A.VerticalFlip(p=0.5) if augment else A.NoOp(),
            A.Rotate(limit=15, p=0.5) if augment else A.NoOp(),
            A.CenterCrop(224, 224),
        ])
        
        self.get_hs_transform = lambda: A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5) if augment else A.NoOp(),
            A.VerticalFlip(p=0.5) if augment else A.NoOp(),
            A.Rotate(limit=15, p=0.5) if augment else A.NoOp(),
            A.CenterCrop(224, 224),
        ])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Load images
        rgb_path = self.rgb_paths[idx]
        rgb = np.array(Image.open(rgb_path).convert('RGB')).astype(np.float32) / 255.0
        ms = tifffile.imread(self.ms_paths[idx]).astype(np.float32)
        hs = tifffile.imread(self.hs_paths[idx]).astype(np.float32)
        label = self.labels[idx]
        
        # Apply transforms
        rgb_transformed = self.rgb_transform(image=rgb)['image']
        
        ms_transformed = self.get_ms_transform()(image=ms)['image']
        hs_transformed = self.get_hs_transform()(image=hs)['image']
        
        # Convert to tensor
        ms_tensor = torch.from_numpy(ms_transformed).permute(2, 0, 1).float()
        hs_tensor = torch.from_numpy(hs_transformed).permute(2, 0, 1).float()
        
        # Normalize
        if ms_tensor.max() > 1.0:
            ms_tensor = ms_tensor / ms_tensor.max()
        if hs_tensor.max() > 1.0:
            hs_tensor = hs_tensor / hs_tensor.max()
        
        # Ensure HS has 125 channels
        if hs_tensor.shape[0] != 125:
            if hs_tensor.shape[0] > 125:
                hs_tensor = hs_tensor[:125]
            else:
                pad = torch.zeros(125 - hs_tensor.shape[0], *hs_tensor.shape[1:])
                hs_tensor = torch.cat([hs_tensor, pad], dim=0)
        
        # Extract RGB handcrafted features (before augmentation)
        if self.use_handcrafted:
            try:
                # Load original image for feature extraction
                img_orig = cv2.imread(rgb_path)
                img_orig_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
                
                # GLCM features (36 dims)
                glcm_feat, _ = RGBFeature.glcm_rgb(rgb_path)
                
                # LBP multiscale features on Green channel (30 dims)
                lbp_feat = RGBFeature.lbp_multiscale(img_orig_rgb[:, :, 1])
                
                # Color features: HSV + ExG + VARI (9 dims)
                color_feat = RGBFeature.rgb_color_veg_features(rgb_path)
                
                # Concatenate all handcrafted features (84 dims total)
                handcrafted = np.concatenate([glcm_feat, lbp_feat, color_feat])
                handcrafted = torch.from_numpy(handcrafted).float()
            except Exception as e:
                # Fallback to zeros if feature extraction fails
                print(f"Warning: Feature extraction failed for {rgb_path}: {e}")
                handcrafted = torch.zeros(84)
        else:
            handcrafted = torch.zeros(84)
        
        # Light augmentation for spectral (only if training)
        if self.augment:
            if torch.rand(1) < 0.3:
                ms_tensor = ms_tensor + torch.randn_like(ms_tensor) * 0.01
                hs_tensor = hs_tensor + torch.randn_like(hs_tensor) * 0.01
            
            if torch.rand(1) < 0.2:
                drop_idx = torch.randint(0, 5, (1,))
                ms_tensor[drop_idx] = 0
                
                num_drop = torch.randint(1, 3, (1,))
                drop_indices = torch.randperm(125)[:num_drop]
                hs_tensor[drop_indices] = 0
        
        ms_tensor = torch.clamp(ms_tensor, 0, 1)
        hs_tensor = torch.clamp(hs_tensor, 0, 1)
        
        return rgb_transformed, ms_tensor, hs_tensor, handcrafted, label


# ==================== TRAINING FUNCTIONS ====================
def train_one_epoch_step1(model, loader, optimizer, criterion, device):
    """Train Step 1 only (Healthy vs Diseased)"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for rgb, ms, hs, handcrafted, labels in loader:
        rgb, ms, hs, handcrafted, labels = rgb.to(device), ms.to(device), hs.to(device), handcrafted.to(device), labels.to(device)
        
        # Convert to binary: 0=Healthy, 1=Diseased
        labels_binary = (labels > 0).long()
        
        optimizer.zero_grad()
        outputs = model(rgb, ms, hs, rgb_handcrafted=handcrafted, step='step1')
        loss = criterion(outputs, labels_binary)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels_binary).sum().item()
    
    return running_loss / len(loader), correct / total


@torch.no_grad()
def validate_step1(model, loader, criterion, device):
    """Validate Step 1"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for rgb, ms, hs, handcrafted, labels in loader:
        rgb, ms, hs, handcrafted, labels = rgb.to(device), ms.to(device), hs.to(device), handcrafted.to(device), labels.to(device)
        labels_binary = (labels > 0).long()
        
        outputs = model(rgb, ms, hs, rgb_handcrafted=handcrafted, step='step1')
        loss = criterion(outputs, labels_binary)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels_binary).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels_binary.cpu().numpy())
    
    return running_loss / len(loader), correct / total, all_preds, all_labels


def train_one_epoch_step2(model, loader, optimizer, criterion, device):
    """Train Step 2 only (Rust vs Other, on diseased samples)"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for rgb, ms, hs, handcrafted, labels in loader:
        # Skip healthy samples
        diseased_mask = labels > 0
        if diseased_mask.sum() == 0:
            continue
        
        rgb = rgb[diseased_mask].to(device)
        ms = ms[diseased_mask].to(device)
        hs = hs[diseased_mask].to(device)
        labels = labels[diseased_mask].to(device)
        # Note: handcrafted features not used in Step 2 (not passed to model)
        
        # Convert to binary: 0=Rust (1), 1=Other (2)
        labels_binary = (labels == 2).long()
        
        optimizer.zero_grad()
        outputs = model(rgb, ms, hs, step='step2')
        loss = criterion(outputs, labels_binary)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels_binary).sum().item()
    
    return running_loss / max(len(loader), 1), correct / max(total, 1)


@torch.no_grad()
def validate_step2(model, loader, criterion, device):
    """Validate Step 2"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for rgb, ms, hs, handcrafted, labels in loader:
        diseased_mask = labels > 0
        if diseased_mask.sum() == 0:
            continue
        
        rgb = rgb[diseased_mask].to(device)
        ms = ms[diseased_mask].to(device)
        hs = hs[diseased_mask].to(device)
        labels = labels[diseased_mask].to(device)
        # Note: handcrafted features not used in Step 2
        
        labels_binary = (labels == 2).long()
        
        outputs = model(rgb, ms, hs, step='step2')
        loss = criterion(outputs, labels_binary)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels_binary).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels_binary.cpu().numpy())
    
    return running_loss / max(len(loader), 1), correct / max(total, 1), all_preds, all_labels


@torch.no_grad()
def validate_final_3class(model, loader, device):
    """Validate final 3-class performance"""
    model.eval()
    all_preds = []
    all_labels = []
    
    for rgb, ms, hs, handcrafted, labels in loader:
        rgb, ms, hs, handcrafted = rgb.to(device), ms.to(device), hs.to(device), handcrafted.to(device)
        
        outputs = model(rgb, ms, hs, rgb_handcrafted=handcrafted, step='both')
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
    
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    return acc, all_preds, all_labels


# ==================== MAIN TRAINING ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 TWO-STEP TRAINING PIPELINE - STAGE 1")
    print("="*70)
    print("Strategy:")
    print("  ✅ Step 1: Healthy vs Diseased (RGB-dominant, 85% RGB)")
    print("  ✅ Step 2: Rust vs Other (Spectral-dominant, 70% Spectral)")
    print("  ✅ Phased training: Step1 → Step2 → Fine-tune")
    print("="*70 + "\n")
    
    # Load data
    data_root = Path(project_root) / 'Data'
    
    print("Loading training data...")
    train_rgb_paths, train_ms_paths, train_hs_paths, train_labels = load_data_from_folder(data_root, 'train')
    print(f"Train samples: {len(train_labels)}")
    print(f"  - Health: {train_labels.count(0)}")
    print(f"  - Rust: {train_labels.count(1)}")
    print(f"  - Other: {train_labels.count(2)}")
    
    print("\nLoading validation data...")
    val_rgb_paths, val_ms_paths, val_hs_paths, val_labels = load_data_from_folder(data_root, 'val')
    
    if len(val_labels) < 50:
        if len(val_labels) > 0:
            print(f"  ⚠️ Only {len(val_labels)} samples in val folder")
        else:
            print("  ⚠️ Val folder is empty")
        
        print("\n📊 Splitting train data: 80% train, 20% validation...")
        from sklearn.model_selection import train_test_split
        train_rgb_paths, val_rgb_paths, train_ms_paths, val_ms_paths, \
        train_hs_paths, val_hs_paths, train_labels, val_labels = train_test_split(
            train_rgb_paths, train_ms_paths, train_hs_paths, train_labels,
            test_size=0.2, random_state=42, stratify=train_labels
        )
        print(f"\nAfter split:")
        print(f"Train samples: {len(train_labels)}")
        print(f"  - Health: {train_labels.count(0)}")
        print(f"  - Rust: {train_labels.count(1)}")
        print(f"  - Other: {train_labels.count(2)}")
    
    print(f"\nVal samples: {len(val_labels)}")
    if len(val_labels) > 0:
        print(f"  - Health: {val_labels.count(0)}")
        print(f"  - Rust: {val_labels.count(1)}")
        print(f"  - Other: {val_labels.count(2)}")
    
    # Create datasets
    train_dataset = Stage1Dataset(
        train_rgb_paths, train_ms_paths, train_hs_paths, train_labels,
        augment=True
    )
    val_dataset = Stage1Dataset(
        val_rgb_paths, val_ms_paths, val_hs_paths, val_labels,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")
    
    # Create model
    model = TwoStepWheatNet(dropout_rate=0.4).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)\n")
    
    # ==================== PHASE 1: TRAIN STEP 1 ====================
    print("\n" + "="*70)
    print("PHASE 1: Training Step 1 (Healthy vs Diseased)")
    print("="*70)
    print("Settings:")
    print("  - Epochs: 20")
    print("  - RGB: 512 channels (frozen ResNet18 + Handcrafted 84 dims)")
    print("  - Spectral: 256 channels (MS+HS encoders, -50% vs old 512)")
    print("  - RGB weight: 65% (Spectral: 35%)")
    print("  - RGB Handcrafted: GLCM(36) + LBP(30) + Color(18) = 84 dims")
    print("  - Loss: Focal Loss (alpha=3.0, gamma=2.0) ← GIẢM TỪ 4.0!")
    print("  - Data balancing: WeightedRandomSampler (Healthy 3x)")
    print("  - Strategy: PRIORITY 1 - Balance alpha + oversample")
    print("="*70 + "\n")
    
    # ===== WEIGHTED SAMPLER: Oversample Healthy 3x =====
    from torch.utils.data import WeightedRandomSampler
    
    print("📊 Creating WeightedRandomSampler...")
    sample_weights = []
    for label in train_labels:
        if label == 0:  # Healthy
            sample_weights.append(2.0)  # Oversample Healthy 2x
        else:  # Diseased (Rust + Other)
            sample_weights.append(1.0)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_labels),
        replacement=True
    )
    
    # Recreate train_loader with sampler
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        sampler=sampler,  # Use sampler instead of shuffle=True
        num_workers=0
    )
    
    print(f"  ✅ Healthy samples will be seen ~3x more often")
    print(f"  ✅ Expected Healthy samples per epoch: {train_labels.count(0) * 3}")
    print(f"  ✅ Expected Diseased samples per epoch: {train_labels.count(1) + train_labels.count(2)}\n")
    
    # Optimizer for Step 1
    optimizer_step1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-3
    )
    
    # Focal Loss with alpha=3.0 (Priority 1)
    print("🎯 Focal Loss settings:")
    print("  - alpha=3.0 (balanced penalty for Healthy errors)")
    print("  - gamma=2.0 (focus on hard samples)")
    print("  - Expected: Healthy Recall 30%→50-60%, Diseased→Healthy 22.5%→12-15%\n")
    criterion_step1 = FocalLoss(alpha=3.0, gamma=2.0)
    
    scheduler_step1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_step1, mode='max', factor=0.5, patience=5
    )
    
    best_step1_acc = 0.0
    
    for epoch in range(1, 21):
        train_loss, train_acc = train_one_epoch_step1(
            model, train_loader, optimizer_step1, criterion_step1, device
        )
        val_loss, val_acc, val_preds, val_labels_list = validate_step1(
            model, val_loader, criterion_step1, device
        )
        
        scheduler_step1.step(val_acc)
        
        print(f"[Phase1 Epoch {epoch:2d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")
        
        if val_acc > best_step1_acc:
            best_step1_acc = val_acc
            # Save checkpoint with metadata
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': val_acc,
                'loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss
            }, 'checkpoints/best_model_step1.pth')
            print(f"  ✓ Saved best Step1 model (Epoch {epoch}, Val Acc: {val_acc:.3f})")
        
        if epoch % 10 == 0:
            print("\nStep 1 Classification Report:")
            print(classification_report(
                val_labels_list, val_preds,
                target_names=['Healthy', 'Diseased'],
                digits=3
            ))
            cm = confusion_matrix(val_labels_list, val_preds)
            print("Confusion Matrix:")
            print("        Pred: Healthy  Diseased")
            print(f"True Healthy : {cm[0][0]:5d} {cm[0][1]:5d}")
            print(f"True Diseased: {cm[1][0]:5d} {cm[1][1]:5d}\n")
    
    # Load best Step 1 model
    checkpoint = torch.load('checkpoints/best_model_step1.pth')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n✅ Phase 1 completed! Best Val Acc: {best_step1_acc:.3f} (Epoch {checkpoint['epoch']})\n")
    else:
        model.load_state_dict(checkpoint)
        print(f"\n✅ Phase 1 completed! Best Val Acc: {best_step1_acc:.3f}\n")
    
    # ==================== PHASE 2: TRAIN STEP 2 ====================
    print("\n" + "="*70)
    print("PHASE 2: Training Step 2 (Rust vs Other)")
    print("="*70)
    print("Settings:")
    print("  - Epochs: 20")
    print("  - RGB weight: 30% (Spectral: 70%)")
    print("  - Only diseased samples")
    print("  - Class weights: [1.2, 1.0] (slight Rust boost)")
    print("="*70 + "\n")
    
    # Freeze Step 1 components
    for param in model.step1_gate.parameters():
        param.requires_grad = False
    for param in model.step1_head.parameters():
        param.requires_grad = False
    
    # Optimizer for Step 2
    optimizer_step2 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-3
    )
    
    # Class weights for Step 2
    class_weights_step2 = torch.FloatTensor([1.0, 1.0]).to(device)
    criterion_step2 = nn.CrossEntropyLoss(weight=class_weights_step2)
    
    scheduler_step2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_step2, mode='max', factor=0.5, patience=5
    )
    
    best_step2_acc = 0.0
    
    for epoch in range(1, 21):
        train_loss, train_acc = train_one_epoch_step2(
            model, train_loader, optimizer_step2, criterion_step2, device
        )
        val_loss, val_acc, val_preds, val_labels_list = validate_step2(
            model, val_loader, criterion_step2, device
        )
        
        scheduler_step2.step(val_acc)
        
        print(f"[Phase2 Epoch {epoch:2d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")
        
        if val_acc > best_step2_acc:
            best_step2_acc = val_acc
            # Save checkpoint with metadata
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': val_acc,
                'loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss
            }, 'checkpoints/best_model_step2.pth')
            print(f"  ✓ Saved best Step2 model (Epoch {epoch}, Val Acc: {val_acc:.3f})")
        
        if epoch % 10 == 0:
            print("\nStep 2 Classification Report:")
            print(classification_report(
                val_labels_list, val_preds,
                target_names=['Rust', 'Other'],
                digits=3
            ))
            cm = confusion_matrix(val_labels_list, val_preds)
            print("Confusion Matrix:")
            print("        Pred: Rust  Other")
            print(f"True Rust : {cm[0][0]:5d} {cm[0][1]:5d}")
            print(f"True Other: {cm[1][0]:5d} {cm[1][1]:5d}\n")
    
    # Load best Step 2 model
    checkpoint = torch.load('checkpoints/best_model_step2.pth')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n✅ Phase 2 completed! Best Val Acc: {best_step2_acc:.3f} (Epoch {checkpoint['epoch']})\n")
    else:
        model.load_state_dict(checkpoint)
        print(f"\n✅ Phase 2 completed! Best Val Acc: {best_step2_acc:.3f}\n")
    
    # ==================== FINAL EVALUATION ====================
    print("\n" + "="*70)
    print("FINAL EVALUATION: 3-Class Performance")
    print("="*70 + "\n")
    
    val_acc, val_preds, val_labels_list = validate_final_3class(model, val_loader, device)
    
    print(f"Final 3-Class Validation Accuracy: {val_acc:.3f}\n")
    print("Classification Report:")
    print(classification_report(
        val_labels_list, val_preds,
        target_names=['Health', 'Rust', 'Other'],
        digits=3
    ))
    
    cm = confusion_matrix(val_labels_list, val_preds)
    print("Confusion Matrix:")
    print("        Pred: Health  Rust  Other")
    for idx, class_name in enumerate(['Health', 'Rust', 'Other']):
        print(f"True {class_name:6s}: {cm[idx][0]:5d} {cm[idx][1]:5d} {cm[idx][2]:5d}")
    
    # Save final model
    torch.save(model.state_dict(), 'checkpoints/best_model_twostep_final.pth')
    
    print("\n" + "="*70)
    print("🎉 Two-Step Training Complete!")
    print(f"Best Step 1 Acc: {best_step1_acc:.3f}")
    print(f"Best Step 2 Acc: {best_step2_acc:.3f}")
    print(f"Final 3-Class Acc: {val_acc:.3f}")
    print("="*70)
