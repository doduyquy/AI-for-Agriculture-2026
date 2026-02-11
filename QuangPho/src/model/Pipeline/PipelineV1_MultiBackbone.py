"""
PipelineV1_MultiBackbone.py - MULTI-BACKBONE FEATURE EXTRACTION

🎯 Strategy:
  - Sử dụng NHIỀU pretrained models để extract RGB features:
    * ResNet18: General features, good baseline
    * EfficientNet-B0: Excellent for small details, efficient
    * DenseNet121: Dense connections, better gradient flow
  
  - Attention-based Fusion:
    * Tự động học weights cho mỗi backbone
    * Adaptive theo input image
    * Better than simple concatenation

✅ Kỳ vọng:
  - Richer RGB features → Better Healthy vs Diseased classification
  - Diseased→Health: 22.5% → 12-15%
  - Overall Accuracy: 67.5% → 73-76%
  - Kaggle Score: 0.64 → 0.72-0.76
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import cv2

# Add project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torchvision.models as models
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Import utilities
try:
    import tifffile
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tifffile"])
    import tifffile

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    print("ERROR: Please install albumentations: pip install albumentations")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image

from src.model.Layer2.AttentionFusion import MultiHeadSpectralAttention
from src.FeatureEngineering.RGBFeature import RGBFeature


# ==================== MULTI-BACKBONE RGB FEATURE EXTRACTOR ====================
class MultiBackboneRGBFeature(nn.Module):
    """
    Multi-backbone RGB feature extractor với attention fusion
    
    Backbones:
      1. ResNet18 (512 dims) - General features
      2. EfficientNet-B0 (1280 dims) - Efficient, detail-oriented
      3. DenseNet121 (1024 dims) - Dense connections
    
    Fusion:
      - Attention-based weighted combination
      - Output: 512 dimensions (same as original)
    """
    def __init__(self, output_dim=512, freeze=True):
        super().__init__()
        
        print("  🔧 Building Multi-Backbone RGB Feature Extractor...")
        
        # ===== Backbone 1: ResNet18 =====
        print("    - ResNet18 (512 features)")
        resnet = models.resnet18(pretrained=True)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])
        self.resnet_dim = 512
        
        # ===== Backbone 2: EfficientNet-B0 =====
        print("    - EfficientNet-B0 (1280 features)")
        efficientnet = models.efficientnet_b0(pretrained=True)
        self.efficient_features = efficientnet.features
        self.efficient_dim = 1280
        
        # ===== Backbone 3: DenseNet121 =====
        print("    - DenseNet121 (1024 features)")
        densenet = models.densenet121(pretrained=True)
        self.densenet_features = densenet.features
        self.densenet_dim = 1024
        
        # Freeze backbones
        if freeze:
            print("    - Freezing all backbones...")
            for param in self.resnet_features.parameters():
                param.requires_grad = False
            for param in self.efficient_features.parameters():
                param.requires_grad = False
            for param in self.densenet_features.parameters():
                param.requires_grad = False
        
        # ===== Attention Fusion =====
        print(f"    - Attention Fusion → {output_dim} dims")
        self.attention_fusion = AttentionFusion(
            input_dims=[self.resnet_dim, self.efficient_dim, self.densenet_dim],
            output_dim=output_dim
        )
        
        self.output_dim = output_dim
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) RGB image
        
        Returns:
            features: (B, 512, 7, 7) fused features
        """
        # Extract from all backbones
        feat_resnet = self.resnet_features(x)  # (B, 512, 7, 7)
        feat_efficient = self.efficient_features(x)  # (B, 1280, H, W)
        feat_densenet = self.densenet_features(x)  # (B, 1024, H, W)
        
        # Ensure same spatial size
        target_size = (7, 7)
        feat_efficient = F.adaptive_avg_pool2d(feat_efficient, target_size)
        feat_densenet = F.adaptive_avg_pool2d(feat_densenet, target_size)
        
        # Attention-based fusion
        fused = self.attention_fusion([feat_resnet, feat_efficient, feat_densenet])
        
        return fused  # (B, 512, 7, 7)


class AttentionFusion(nn.Module):
    """
    Attention-based multi-backbone fusion
    
    Learns to weight each backbone based on input
    Example: Green images → ResNet weight↑, Complex texture → DenseNet weight↑
    """
    def __init__(self, input_dims, output_dim):
        super().__init__()
        self.num_backbones = len(input_dims)
        
        # Project each backbone to output_dim
        self.projections = nn.ModuleList([
            nn.Conv2d(dim, output_dim, kernel_size=1)
            for dim in input_dims
        ])
        
        # Attention network: learns weights for each backbone
        total_dim = sum(input_dims)
        self.attention = nn.Sequential(
            nn.Conv2d(total_dim, output_dim // 2, kernel_size=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim // 2, self.num_backbones, kernel_size=1),
            nn.Softmax(dim=1)  # Softmax over backbones
        )
    
    def forward(self, feature_list):
        """
        Args:
            feature_list: [feat1, feat2, feat3] each (B, C_i, H, W)
        
        Returns:
            fused: (B, output_dim, H, W)
        """
        # Concatenate for attention
        concat = torch.cat(feature_list, dim=1)
        
        # Compute attention weights (B, num_backbones, H, W)
        weights = self.attention(concat)
        
        # Project and weight
        fused = torch.zeros_like(self.projections[0](feature_list[0]))
        for i, (proj, feat) in enumerate(zip(self.projections, feature_list)):
            projected = proj(feat)
            fused = fused + weights[:, i:i+1] * projected
        
        return fused


# ==================== FOCAL LOSS ====================
class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    def __init__(self, alpha=3.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ==================== ASYMMETRIC BIASED GATE ====================
class AsymmetricBiasedGate(nn.Module):
    """Asymmetric gate for RGB (512) + Spectral (256) fusion"""
    def __init__(self, rgb_channels=512, spec_channels=256, rgb_weight=0.85):
        super().__init__()
        self.rgb_weight = rgb_weight
        self.spec_weight = 1.0 - rgb_weight
        
        total_channels = rgb_channels + spec_channels
        hidden_channels = (rgb_channels + spec_channels) // 2
        
        self.gate = nn.Sequential(
            nn.Conv2d(total_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid()
        )
        
        if spec_channels != rgb_channels:
            self.spec_proj = nn.Conv2d(spec_channels, rgb_channels, 1)
        else:
            self.spec_proj = nn.Identity()
    
    def forward(self, f_rgb, f_spec):
        concat = torch.cat([f_rgb, f_spec], dim=1)
        gate = self.gate(concat)
        
        f_spec_proj = self.spec_proj(f_spec)
        fused = self.rgb_weight * f_rgb + self.spec_weight * gate * f_spec_proj
        
        return fused


# ==================== TWO-STEP WHEAT NET (MULTI-BACKBONE) ====================
class MultiBackboneTwoStepWheatNet(nn.Module):
    """
    TwoStepWheatNet với Multi-Backbone RGB Feature Extractor
    
    RGB: ResNet18 + EfficientNet + DenseNet (attention fusion)
    Spectral: MS + HS encoders (same as original)
    """
    def __init__(self, dropout_rate=0.4):
        super().__init__()
        
        print("\n🏗️ Building MultiBackbone TwoStepWheatNet...")
        
        # ===== RGB: Multi-Backbone =====
        self.rgb_backbone = MultiBackboneRGBFeature(output_dim=512, freeze=True)
        
        # ===== MS Encoder =====
        print("  1. MS Encoder (5→128 channels)...")
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
        
        # ===== HS Encoder =====
        print("  2. HS Encoder (125→128 channels)...")
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
        
        # ===== Spectral Attention =====
        print("  3. Spectral Attention Fusion...")
        self.spectral_attention = MultiHeadSpectralAttention(
            ms_channels=128, hs_channels=128, num_heads=4, dropout=dropout_rate
        )
        
        self.spectral_proj = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        
        # ===== RGB Handcrafted Features =====
        print("  4. RGB Handcrafted Features (84→128)...")
        self.rgb_handcrafted_proj = nn.Sequential(
            nn.Linear(84, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128)
        )
        
        self.rgb_fusion = nn.Linear(512 + 128, 512)
        
        # ===== Step 1: Healthy vs Diseased =====
        print("  5. Step 1 (Healthy vs Diseased, 65% RGB)...")
        self.step1_gate = AsymmetricBiasedGate(512, 256, rgb_weight=0.85)
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
            nn.Linear(128, 2)
        )
        
        # ===== Step 2: Rust vs Other =====
        print("  6. Step 2 (Rust vs Other, 70% Spectral)...")
        self.step2_gate = AsymmetricBiasedGate(512, 256, rgb_weight=0.30)
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
            nn.Linear(128, 2)
        )
        
        print("✅ Multi-Backbone Model built!\n")
    
    def forward(self, rgb, ms, hs, rgb_handcrafted=None, step='both'):
        # RGB features (MULTI-BACKBONE!)
        with torch.no_grad():
            f_rgb_cnn = self.rgb_backbone(rgb)
        
        # Spectral features
        f_ms = self.ms_encoder(ms)
        f_hs = self.hs_encoder(hs)
        f_spec = self.spectral_attention(f_ms, f_hs)
        f_spec = self.spectral_proj(f_spec)
        
        # Enhance RGB with handcrafted
        if rgb_handcrafted is not None and step in ['step1', 'both']:
            f_rgb_pooled = F.adaptive_avg_pool2d(f_rgb_cnn, (1, 1)).flatten(1)
            f_handcrafted = self.rgb_handcrafted_proj(rgb_handcrafted)
            f_rgb_fused = torch.cat([f_rgb_pooled, f_handcrafted], dim=1)
            f_rgb_fused = self.rgb_fusion(f_rgb_fused)
            f_rgb = f_rgb_fused.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 7, 7)
        else:
            f_rgb = f_rgb_cnn
        
        if step == 'step1':
            f_step1 = self.step1_gate(f_rgb, f_spec)
            return self.step1_head(f_step1)
        
        elif step == 'step2':
            f_step2 = self.step2_gate(f_rgb, f_spec)
            return self.step2_head(f_step2)
        
        else:  # both
            f_step1 = self.step1_gate(f_rgb, f_spec)
            step1_logits = self.step1_head(f_step1)
            
            f_step2 = self.step2_gate(f_rgb, f_spec)
            step2_logits = self.step2_head(f_step2)
            
            step1_probs = F.softmax(step1_logits, dim=1)
            step2_probs = F.softmax(step2_logits, dim=1)
            
            p_healthy = step1_probs[:, 0].unsqueeze(1)
            p_rust = (step1_probs[:, 1] * step2_probs[:, 0]).unsqueeze(1)
            p_other = (step1_probs[:, 1] * step2_probs[:, 1]).unsqueeze(1)
            
            final_probs = torch.cat([p_healthy, p_rust, p_other], dim=1)
            return torch.log(final_probs + 1e-8)


# ==================== DATA LOADING & DATASET ====================
def load_data_from_folder(data_root, split='train'):
    rgb_folder = data_root / split / 'RGB'
    ms_folder = data_root / split / 'MS'
    hs_folder = data_root / split / 'HS'
    
    if not rgb_folder.exists():
        return [], [], [], []
    
    rgb_paths, ms_paths, hs_paths, labels = [], [], [], []
    rgb_files = sorted(list(rgb_folder.glob('*.png')) + list(rgb_folder.glob('*.tif')))
    
    for rgb_file in rgb_files:
        filename = rgb_file.name
        stem = rgb_file.stem
        
        if filename.startswith('Health'):
            label = 0
        elif filename.startswith('Rust'):
            label = 1
        elif filename.startswith('Other'):
            label = 2
        else:
            continue
        
        ms_file = ms_folder / f"{stem}.tif"
        hs_file = hs_folder / f"{stem}.tif"
        
        if ms_file.exists() and hs_file.exists():
            rgb_paths.append(str(rgb_file))
            ms_paths.append(str(ms_file))
            hs_paths.append(str(hs_file))
            labels.append(label)
    
    return rgb_paths, ms_paths, hs_paths, labels


class Stage1Dataset(torch.utils.data.Dataset):
    def __init__(self, rgb_paths, ms_paths, hs_paths, labels, augment=False):
        self.rgb_paths = rgb_paths
        self.ms_paths = ms_paths
        self.hs_paths = hs_paths
        self.labels = labels
        self.augment = augment
        
        if augment:
            self.rgb_transform = A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
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
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        rgb_path = self.rgb_paths[idx]
        rgb = np.array(Image.open(rgb_path).convert('RGB')).astype(np.float32) / 255.0
        ms = tifffile.imread(self.ms_paths[idx]).astype(np.float32)
        hs = tifffile.imread(self.hs_paths[idx]).astype(np.float32)
        label = self.labels[idx]
        
        rgb_transformed = self.rgb_transform(image=rgb)['image']
        
        ms_tensor = torch.from_numpy(ms).permute(2, 0, 1).float()
        hs_tensor = torch.from_numpy(hs).permute(2, 0, 1).float()
        
        if ms_tensor.shape[1] != 224 or ms_tensor.shape[2] != 224:
            ms_tensor = F.interpolate(ms_tensor.unsqueeze(0), size=(224, 224), mode='bilinear')[0]
        if hs_tensor.shape[1] != 224 or hs_tensor.shape[2] != 224:
            hs_tensor = F.interpolate(hs_tensor.unsqueeze(0), size=(224, 224), mode='bilinear')[0]
        
        if ms_tensor.max() > 1.0:
            ms_tensor = ms_tensor / ms_tensor.max()
        if hs_tensor.max() > 1.0:
            hs_tensor = hs_tensor / hs_tensor.max()
        
        if hs_tensor.shape[0] != 125:
            if hs_tensor.shape[0] > 125:
                hs_tensor = hs_tensor[:125]
            else:
                pad = torch.zeros(125 - hs_tensor.shape[0], *hs_tensor.shape[1:])
                hs_tensor = torch.cat([hs_tensor, pad], dim=0)
        
        try:
            img_orig = cv2.imread(rgb_path)
            img_orig_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
            
            glcm_feat, _ = RGBFeature.glcm_rgb(rgb_path)
            lbp_feat = RGBFeature.lbp_multiscale(img_orig_rgb[:, :, 1])
            color_feat = RGBFeature.rgb_color_veg_features(rgb_path)
            
            handcrafted = np.concatenate([glcm_feat, lbp_feat, color_feat])
            handcrafted = torch.from_numpy(handcrafted).float()
        except:
            handcrafted = torch.zeros(84)
        
        return rgb_transformed, ms_tensor, hs_tensor, handcrafted, label


# ==================== TRAINING FUNCTIONS ====================
def train_one_epoch_step1(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for rgb, ms, hs, handcrafted, labels in loader:
        rgb, ms, hs, handcrafted, labels = rgb.to(device), ms.to(device), hs.to(device), handcrafted.to(device), labels.to(device)
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
def validate_step1(model, loader, criterion, device, healthy_threshold=0.5):
    """
    Args:
        healthy_threshold: If prob(Healthy) >= threshold, predict Healthy
                          Lower value → More Healthy predictions
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for rgb, ms, hs, handcrafted, labels in loader:
        rgb, ms, hs, handcrafted = rgb.to(device), ms.to(device), hs.to(device), handcrafted.to(device)
        labels_binary = (labels > 0).long()
        
        outputs = model(rgb, ms, hs, rgb_handcrafted=handcrafted, step='step1')
        loss = criterion(outputs, labels_binary)
        
        running_loss += loss.item()
        
        # Threshold-based: Lower threshold → More Healthy predictions
        probs = F.softmax(outputs, dim=1)
        predicted = (probs[:, 0] >= healthy_threshold).long()  # 1 if Healthy
        predicted = 1 - predicted  # Flip: 0=Healthy, 1=Diseased
        
        total += labels.size(0)
        correct += predicted.cpu().eq(labels_binary).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels_binary.numpy())
    
    return running_loss / len(loader), correct / total, all_preds, all_labels


# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 MULTI-BACKBONE TWO-STEP TRAINING")
    print("="*70)
    print("RGB: ResNet18 + EfficientNet-B0 + DenseNet121 (Attention Fusion)")
    print("Spectral: MS + HS (Multi-Head Attention)")
    print("="*70 + "\n")
    
    # Load data
    data_root = Path(project_root) / 'Data'
    
    train_rgb_paths, train_ms_paths, train_hs_paths, train_labels = load_data_from_folder(data_root, 'train')
    val_rgb_paths, val_ms_paths, val_hs_paths, val_labels = load_data_from_folder(data_root, 'val')
    
    if len(val_labels) < 50:
        train_rgb_paths, val_rgb_paths, train_ms_paths, val_ms_paths, \
        train_hs_paths, val_hs_paths, train_labels, val_labels = train_test_split(
            train_rgb_paths, train_ms_paths, train_hs_paths, train_labels,
            test_size=0.2, random_state=42, stratify=train_labels
        )
    
    print(f"Train: {len(train_labels)} samples")
    print(f"Val: {len(val_labels)} samples\n")
    
    # Create datasets
    train_dataset = Stage1Dataset(train_rgb_paths, train_ms_paths, train_hs_paths, train_labels, augment=True)
    val_dataset = Stage1Dataset(val_rgb_paths, val_ms_paths, val_hs_paths, val_labels, augment=False)
    
    # Weighted sampler (oversample Healthy 5x to improve recall)
    sample_weights = [5.0 if label == 0 else 1.0 for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Model
    model = MultiBackboneTwoStepWheatNet(dropout_rate=0.4).to(device)
    
    # Optimizer & Loss
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-3
    )
    criterion = FocalLoss(alpha=3.0, gamma=2.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Train Step 1
    print("="*70)
    print("PHASE 1: Training Step 1 (Healthy vs Diseased)")
    print("Strategy: Low threshold (0.35) → Increase Healthy recall")
    print("          Healthy oversample 5x → Better Healthy learning")
    print("="*70 + "\n")
    
    healthy_threshold = 0.35  # Lower = more Healthy predictions
    best_acc = 0.0
    for epoch in range(1, 21):
        train_loss, train_acc = train_one_epoch_step1(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels_list = validate_step1(model, val_loader, criterion, device, healthy_threshold)
        
        scheduler.step(val_acc)
        
        print(f"[Epoch {epoch:2d}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/multibackbone_step1.pth')
            print(f"  ✅ Saved! Best Acc: {best_acc:.3f}")
        
        if epoch % 10 == 0:
            print("\nClassification Report:")
            print(classification_report(val_labels_list, val_preds, target_names=['Healthy', 'Diseased'], digits=3))
            cm = confusion_matrix(val_labels_list, val_preds)
            print("Confusion Matrix:")
            print(f"          Healthy  Diseased")
            print(f"Healthy : {cm[0][0]:5d} {cm[0][1]:5d}")
            print(f"Diseased: {cm[1][0]:5d} {cm[1][1]:5d}\n")
    
    print(f"\n✅ Step 1 Training Complete! Best Acc: {best_acc:.3f}")
    print("="*70 + "\n")
