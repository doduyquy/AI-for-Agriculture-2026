"""
PipelineV1_Stage1.py - GIAI ĐOẠN 1: ÍT DATA (200 samples/class)

🎯 Mục tiêu:
  - Train ổn định
  - Tránh overfitting
  - Kiểm chứng pipeline

🔧 Cấu hình:
  - Data split: 80% train, 20% val
  - Augmentation: Nhẹ nhàng (Flip, Rotate ±15°, Color jitter nhẹ)
  - Loss: Weighted CrossEntropyLoss (KHÔNG dùng Focal Loss)
  - Kiến trúc: BỎ Soft Competition, dùng pipeline đơn giản
  - Freeze: ResNet18 backbone
  - Train: MS/HS encoder, Attention, Gate, Classifier
"""

import sys
import os
import torch
import torch.nn as nn
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

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model.Layer1.RGBExaction import WheatRGBDataset
from src.model.Layer1.MSExaction import MSFeatureExtractor
from src.model.Layer1.HSExaction import HSFeatureExtractor
from src.model.Layer1.CNNRGB import RGBCNNFeature

from src.model.Layer2.AttentionFusion import MultiHeadSpectralAttention
from src.model.OutputLayer.OutputLayer import ContextGatedFusion, RobustWheatHead
from src.Preprocessing.PreRGB import cnn_transform


# ==================== SIMPLIFIED MODEL FOR STAGE 1 ====================
class WheatNetStage1(nn.Module):
    """
    Simplified architecture for Stage 1 (limited data):
    
    RGB → CNN (FROZEN) → F_rgb
    MS  → Encoder → F_ms
    HS  → Encoder → F_hs
    
    [F_ms + F_hs] → AttentionFusion → F_spec
    F_rgb + F_spec → ContextGatedFusion → Head
    
    ❌ NO Soft Competition (requires more data)
    ✅ FREEZE ResNet18 backbone
    ✅ Only train: MS/HS encoders, Attention, Gate, Classifier
    """
    def __init__(self, num_classes=3, dropout_rate=0.4):
        super().__init__()
        
        # ===== RGB Branch - ResNet18 FROZEN =====
        self.rgb_backbone = RGBCNNFeature(backbone='resnet18', pretrained=True)
        
        # 🔒 FREEZE ResNet18 backbone
        print("\n🔒 Freezing ResNet18 backbone...")
        for param in self.rgb_backbone.parameters():
            param.requires_grad = False
        print("   ✓ ResNet18 frozen - will NOT be trained")
        
        # RGB produces: (batch, 512, 7, 7)
        
        # ===== MS Branch (5 bands) - TRAINABLE =====
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
            nn.AdaptiveAvgPool2d((7, 7))  # Match RGB spatial size
        )
        
        # ===== HS Branch (125 bands) - TRAINABLE =====
        self.hs_encoder = nn.Sequential(
            nn.Conv2d(125, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))  # Match RGB spatial size
        )
        
        # ===== Spectral Fusion (MS + HS) - TRAINABLE =====
        # MS: 128 channels, HS: 128 channels
        self.spectral_attention = MultiHeadSpectralAttention(
            ms_channels=128,
            hs_channels=128,
            num_heads=4,
            dropout=dropout_rate
        )
        # Output: 256 channels (128 MS + 128 HS)
        
        # Project spectral features to match RGB
        self.spectral_proj = nn.Sequential(
            nn.Conv2d(256, 512, 1),  # 256 → 512 to match RGB
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        
        # ===== RGB + Spectral Fusion - TRAINABLE =====
        # ContextGatedFusion requires same channels for both inputs
        self.rgb_spectral_gate = ContextGatedFusion(channels=512)
        # Output: 512 channels
        
        # ===== Classification Head - TRAINABLE =====
        # Head does global pooling internally
        self.head = RobustWheatHead(
            channels=512,
            num_classes=num_classes,
            dropout_rate=dropout_rate + 0.1
        )
        
    def forward(self, rgb, ms, hs):
        # RGB features (FROZEN - no gradient)
        with torch.no_grad():
            f_rgb = self.rgb_backbone(rgb)  # (batch, 512, 7, 7)
        
        # MS features (TRAINABLE)
        f_ms = self.ms_encoder(ms)  # (batch, 128, 7, 7)
        
        # HS features (TRAINABLE)
        f_hs = self.hs_encoder(hs)  # (batch, 128, 7, 7)
        
        # Spectral fusion (MS + HS) via Attention
        f_spec = self.spectral_attention(f_ms, f_hs)  # (batch, 256, 7, 7)
        f_spec = self.spectral_proj(f_spec)  # (batch, 512, 7, 7)
        
        # RGB + Spectral fusion via Context Gate
        f_fused = self.rgb_spectral_gate(f_rgb, f_spec)  # (batch, 512, 7, 7)
        
        # Classification (head does global pooling internally)
        logits = self.head(f_fused)  # (batch, num_classes)
        
        return logits


# ==================== DATA LOADING ====================
def load_data_from_folder(data_root, split='train'):
    """
    Load RGB, MS, HS paths and labels from folder
    
    Folder structure:
    Data/train/RGB/Health_hyper_1.png, Rust_hyper_1.png, Other_hyper_1.png
    Data/train/MS/Health_hyper_1.tif, Rust_hyper_1.tif, Other_hyper_1.tif
    Data/train/HS/Health_hyper_1.tif, Rust_hyper_1.tif, Other_hyper_1.tif
    """
    rgb_folder = data_root / split / 'RGB'
    ms_folder = data_root / split / 'MS'
    hs_folder = data_root / split / 'HS'
    
    if not rgb_folder.exists():
        return [], [], [], []
    
    rgb_paths, ms_paths, hs_paths, labels = [], [], [], []
    
    # Get all RGB files (*.png or *.tif)
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
            continue  # Skip unknown files
        
        # Find corresponding MS and HS files
        ms_file = ms_folder / f"{stem}.tif"
        hs_file = hs_folder / f"{stem}.tif"
        
        if ms_file.exists() and hs_file.exists():
            rgb_paths.append(str(rgb_file))
            ms_paths.append(str(ms_file))
            hs_paths.append(str(hs_file))
            labels.append(label)
    
    return rgb_paths, ms_paths, hs_paths, labels


# ==================== DATASET WITH LIGHT AUGMENTATION ====================
class Stage1Dataset(torch.utils.data.Dataset):
    """
    Dataset for Stage 1 with LIGHT augmentation
    
    Augmentation strategy:
    - RGB: Flip, Rotate ±15°, Color jitter nhẹ
    - MS/HS: Gaussian noise nhỏ, Band dropout (1-2 bands)
    - NO heavy augmentation (dataset too small)
    """
    def __init__(self, rgb_paths, ms_paths, hs_paths, labels, augment=False):
        self.rgb_paths = rgb_paths
        self.ms_paths = ms_paths
        self.hs_paths = hs_paths
        self.labels = labels
        self.augment = augment
        
        # ===== LIGHT Augmentation for Stage 1 =====
        if augment:
            # RGB augmentation - MODERATE
            self.rgb_transform = A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),  # ±15° only
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,
                    contrast_limit=0.15,
                    p=0.3
                ),
                A.CenterCrop(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            # MS/HS augmentation - SEPARATE (they may have different sizes)
            self.ms_transform = A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.CenterCrop(224, 224),
            ])
            
            self.hs_transform = A.Compose([
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.CenterCrop(224, 224),
            ])
        else:
            # Validation - NO augmentation
            self.rgb_transform = A.Compose([
                A.Resize(256, 256),
                A.CenterCrop(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            self.ms_transform = A.Compose([
                A.Resize(256, 256),
                A.CenterCrop(224, 224),
            ])
            
            self.hs_transform = A.Compose([
                A.Resize(256, 256),
                A.CenterCrop(224, 224),
            ])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Load images
        # RGB: Use cv2 for PNG/TIFF support
        rgb_path = self.rgb_paths[idx]
        if rgb_path.endswith('.png') or rgb_path.endswith('.jpg'):
            rgb = cv2.imread(rgb_path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        else:
            rgb = tifffile.imread(rgb_path)
        
        # MS and HS: Always TIFF
        ms = tifffile.imread(self.ms_paths[idx])
        hs = tifffile.imread(self.hs_paths[idx])
        label = self.labels[idx]
        
        # Ensure RGB is 3-channel
        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        elif rgb.shape[-1] != 3:
            rgb = rgb[..., :3]
        
        # Normalize RGB to [0, 1] if needed
        if rgb.max() > 1.0:
            rgb = rgb.astype(np.float32) / 255.0
        
        # Apply RGB transform
        rgb_transformed = self.rgb_transform(image=rgb)['image']
        
        # Apply MS transform
        ms_transformed = self.ms_transform(image=ms)['image']
        
        # Apply HS transform
        hs_transformed = self.hs_transform(image=hs)['image']
        
        # Convert MS/HS to tensor and normalize
        ms_tensor = torch.from_numpy(ms_transformed).permute(2, 0, 1).float()
        hs_tensor = torch.from_numpy(hs_transformed).permute(2, 0, 1).float()
        
        # ===== FIX: Ensure HS always has 125 bands =====
        if hs_tensor.shape[0] > 125:
            hs_tensor = hs_tensor[:125, :, :]  # Crop to 125 bands
        elif hs_tensor.shape[0] < 125:
            # Pad to 125 bands
            padding = torch.zeros(125 - hs_tensor.shape[0], hs_tensor.shape[1], hs_tensor.shape[2])
            hs_tensor = torch.cat([hs_tensor, padding], dim=0)
        
        # Normalize MS/HS to [0, 1]
        if ms_tensor.max() > 1.0:
            ms_tensor = ms_tensor / ms_tensor.max()
        if hs_tensor.max() > 1.0:
            hs_tensor = hs_tensor / hs_tensor.max()
        
        # LIGHT spectral augmentation (only on training)
        if self.augment:
            # Gaussian noise (very small)
            if torch.rand(1) < 0.3:
                ms_tensor = ms_tensor + torch.randn_like(ms_tensor) * 0.01
                hs_tensor = hs_tensor + torch.randn_like(hs_tensor) * 0.01
            
            # Band dropout (1-2 bands randomly)
            if torch.rand(1) < 0.2:
                # MS: drop 1 band out of 5
                drop_idx = torch.randint(0, 5, (1,))
                ms_tensor[drop_idx] = 0
                
                # HS: drop 1-2 bands out of 125
                num_drop = torch.randint(1, 3, (1,))
                drop_indices = torch.randperm(125)[:num_drop]
                hs_tensor[drop_indices] = 0
        
        # Clamp to valid range
        ms_tensor = torch.clamp(ms_tensor, 0, 1)
        hs_tensor = torch.clamp(hs_tensor, 0, 1)
        
        return rgb_transformed, ms_tensor, hs_tensor, label


# ==================== TRAINING FUNCTIONS ====================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for rgb, ms, hs, labels in loader:
        rgb, ms, hs, labels = rgb.to(device), ms.to(device), hs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(rgb, ms, hs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for rgb, ms, hs, labels in loader:
        rgb, ms, hs, labels = rgb.to(device), ms.to(device), hs.to(device), labels.to(device)
        
        outputs = model(rgb, ms, hs)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(loader), correct / total, all_preds, all_labels


# ==================== MAIN TRAINING ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 STAGE 1 TRAINING PIPELINE - LIMITED DATA")
    print("="*70)
    print("Strategy:")
    print("  ✅ Simple architecture (NO Soft Competition)")
    print("  ✅ Freeze ResNet18 backbone")
    print("  ✅ Light augmentation (Flip, Rotate ±15°)")
    print("  ✅ Weighted CrossEntropyLoss")
    print("  ✅ Train only: MS/HS encoders, Attention, Gate, Classifier")
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
    
    # Split from training data if validation is insufficient
    if len(val_labels) < 50:
        if len(val_labels) > 0:
            print(f"  ⚠️ Only {len(val_labels)} samples in val folder (too few)")
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
    print("\n✅ Using Stage1Dataset with LIGHT augmentation")
    print("   Train: Flip, Rotate ±15°, Color jitter (mild)")
    print("   Val: NO augmentation (evaluation only)\n")
    
    train_dataset = Stage1Dataset(
        train_rgb_paths, train_ms_paths, train_hs_paths, train_labels,
        augment=True
    )
    val_dataset = Stage1Dataset(
        val_rgb_paths, val_ms_paths, val_hs_paths, val_labels,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=0
    )
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create model
    model = WheatNetStage1(num_classes=3, dropout_rate=0.4).to(device)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"📊 Model parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"   Frozen: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    print(f"   → ResNet18 frozen, only training spectral encoders + fusion!\n")
    
    # Optimizer - only for trainable parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-3
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # ===== BALANCED CLASS WEIGHTS =====
    print("📊 Computing balanced class weights from training data...")
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print(f"Balanced class weights:")
    print(f"  Health (0): {class_weights[0]:.3f}")
    print(f"  Rust   (1): {class_weights[1]:.3f}")
    print(f"  Other  (2): {class_weights[2]:.3f}")
    print(f"  Strategy: Let sklearn compute balanced weights naturally\n")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Training loop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    print("="*70)
    print("Starting training...")
    print(f"  - Max epochs: 100")
    print(f"  - Learning rate: 1e-4")
    print(f"  - Weight decay: 1e-3")
    print(f"  - Dropout: 0.4 (encoders), 0.5 (head)")
    print(f"  - Loss: Weighted CrossEntropyLoss (balanced)")
    print(f"  - Early stopping patience: {patience}")
    print(f"  - Frozen: ResNet18 backbone")
    print(f"  - Trainable: MS/HS encoders, Attention, Gate, Classifier")
    print("="*70 + "\n")
    
    for epoch in range(1, 101):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_preds, val_labels_list = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Print progress
        print(f"[Epoch {epoch:3d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, 'checkpoints/best_model_stage1.pth')
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.3f}, Val Loss: {val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement for {patience_counter}/{patience} epochs")
        
        # Classification report every 10 epochs
        if epoch % 10 == 0:
            print("\n" + "="*60)
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
            print("="*60 + "\n")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹ Early stopping triggered at epoch {epoch}")
            print(f"Best Val Acc: {best_val_acc:.3f} at epoch {epoch - patience_counter}")
            break
    
    print("\n" + "="*70)
    print("🎉 Training completed!")
    print(f"Best Val Acc: {best_val_acc:.3f}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print("="*70)
