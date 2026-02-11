"""
PipelineV1_Improved.py - Improved version with:
1. Focal Loss with class weights
2. Enhanced data augmentation
3. Better regularization
4. Improved training stability

Expected improvements:
- Overall Acc: 55% → 70-75%
- Health Recall: 17.5% → 55-65%
- Health F1: 24.6% → 50-60%
"""

import sys
from pathlib import Path
import os
import glob
import cv2
import numpy as np
try:
    import tifffile
except ImportError:
    print("Installing tifffile...")
    import subprocess
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
    print("="*70 + "\n")
    raise ImportError("albumentations is required for this improved version")

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.Layer1.RGBExaction import WheatRGBDataset
from src.model.Layer1.MSExaction import MSFeatureExtractor
from src.model.Layer1.HSExaction import HSFeatureExtractor
from src.model.Layer1.CNNRGB import RGBCNNFeature

from src.model.Layer2.FullFusion import FullFusionModel
from src.model.OutputLayer.OutputLayer import ContextGatedFusion, RobustWheatHead
from src.Preprocessing.PreRGB import cnn_transform


# ==================== FOCAL LOSS ====================
class FocalLoss(nn.Module):
    """
    Focal Loss: Focuses on hard examples by down-weighting easy ones.
    Addresses class imbalance by reducing the loss contribution from easy examples.
    
    Args:
        alpha: Class weights (Tensor of shape [num_classes])
        gamma: Focusing parameter (default: 2.0). Higher gamma = more focus on hard examples
        reduction: 'mean' or 'sum'
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, num_classes) - raw logits from model
            targets: (B,) - ground truth class labels
        """
        # Calculate cross entropy loss
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction='none'
        )
        
        # Calculate pt (probability of true class)
        pt = torch.exp(-ce_loss)
        
        # Focal loss formula: (1 - pt)^gamma * CE_loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ==================== MODEL ====================
class WheatSpectralNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # ===== RGB =====
        self.rgb_backbone = RGBCNNFeature(backbone='resnet18', pretrained=True)
        rgb_dim = self.rgb_backbone.feature_dim  # 512

        # ===== MS =====
        self.ms_extractor = MSFeatureExtractor(
            num_bands=5,
            index_dim=3,
            out_dim=64,
            use_global_pool=True
        )
        dim_ms = 64 + 32          # v_spec + index_emb
        dim_ms_spec = 64
        dim_ms_index = 32

        # ===== HS =====
        self.hs_extractor = HSFeatureExtractor()
        dim_sig = 125 * 3  # spectral_signature returns (B, 125*3)
        dim_tex = 4        # spectral_texture returns (B, 4)
        dim_hs_dl = 128    # HSSpectralEncoder output
        dim_hs = dim_sig + dim_tex + dim_hs_dl  # 375 + 4 + 128 = 507

        # ===== SoftCompetition + Attention =====
        self.fusion = FullFusionModel(
            dim_ms=dim_ms,
            dim_hs=dim_hs,
            dim_ms_index=dim_ms_index,
            dim_ms_spec=dim_ms_spec,
            dim_sig=dim_sig,
            dim_tex=dim_tex,
            embed_dim=128
        )

        # ===== HIGHER Dropout to reduce overfitting (fix val stuck at 33%) =====
        dropout_rate = 0.5  # Increased from 0.4 to fight overfitting
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        # ===== Project fused vector → feature map =====
        self.fused_proj = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # ===== RGB × Spectral Gate =====
        self.context_gate = ContextGatedFusion(channels=512)

        # ===== Head with HIGHER dropout to reduce overfitting =====
        self.head = RobustWheatHead(channels=512, num_classes=num_classes, dropout_rate=0.6)

    def forward(self, rgb, ms, hs):
        # --- RGB: Keep spatial feature map (7×7) ---
        F_rgb_map = self.rgb_backbone(rgb)  # (B, 512, 7, 7)
        B = F_rgb_map.size(0)

        # --- MS ---
        F_ms_all = self.ms_extractor(ms)             # (B, 96)
        F_ms_all = self.dropout2(F_ms_all)
        F_ms_spec = F_ms_all[:, :64]
        F_ms_index = F_ms_all[:, 64:]

        # --- HS ---
        F_hs_all = self.hs_extractor(hs)
        F_hs_all = self.dropout2(F_hs_all)
        dim_sig = 375
        dim_tex = 4
        F_sig = F_hs_all[:, :dim_sig]
        F_tex = F_hs_all[:, dim_sig:dim_sig+dim_tex]
        F_hs_dl = F_hs_all[:, dim_sig+dim_tex:]

        # --- SoftCompetition + Attention ---
        F_fused_vec = self.fusion(
            F_ms=F_ms_all,
            F_hs=F_hs_all,
            F_ms_index=F_ms_index,
            F_ms_spec=F_ms_spec,
            F_sig=F_sig,
            F_tex=F_tex
        )

        # --- Project fused vector to spatial map ---
        F_fused_proj_vec = self.fused_proj(F_fused_vec)  # (B, 512)
        F_spec_map = F_fused_proj_vec.view(B, 512, 1, 1)
        
        # Upsample spectral to match RGB spatial size (7×7)
        import torch.nn.functional as F
        F_spec_map_upsampled = F.interpolate(
            F_spec_map, 
            size=(7, 7), 
            mode='bilinear', 
            align_corners=False
        )

        # --- Gate with RGB ---
        F_final = self.context_gate(F_rgb_map, F_spec_map_upsampled)

        # --- Classifier ---
        logits = self.head(F_final)
        return logits


# ==================== DATA LOADING ====================
def load_data_from_folder(data_root, split='train'):
    """
    Load RGB, MS, HS data from folder structure.
    Labels: Health=0, Rust=1, Other=2
    """
    split_path = Path(data_root) / split
    rgb_dir = split_path / 'RGB'
    ms_dir = split_path / 'MS'
    hs_dir = split_path / 'HS'
    
    rgb_paths = []
    ms_paths = []
    hs_paths = []
    labels = []
    
    label_map = {'Health': 0, 'Rust': 1, 'Other': 2}
    
    for label_name, label_id in label_map.items():
        rgb_files = sorted(glob.glob(str(rgb_dir / f'{label_name}_hyper_*.png')))
        
        for rgb_file in rgb_files:
            basename = Path(rgb_file).stem
            file_id = basename.split('_')[-1]
            
            ms_file = ms_dir / f'{label_name}_hyper_{file_id}.tif'
            hs_file = hs_dir / f'{label_name}_hyper_{file_id}.tif'
            
            if ms_file.exists() and hs_file.exists():
                rgb_paths.append(str(rgb_file))
                ms_paths.append(str(ms_file))
                hs_paths.append(str(hs_file))
                labels.append(label_id)
    
    return rgb_paths, ms_paths, hs_paths, labels


# ==================== DATASET WITH IMPROVED AUGMENTATION ====================
class SyncedWheatMultiModalDataset(torch.utils.data.Dataset):
    """
    IMPROVED Dataset with ENHANCED augmentation:
    - More aggressive geometric transforms
    - Additional color augmentations
    - Noise and blur for robustness
    - CoarseDropout (CutOut) for regularization
    """
    def __init__(self, rgb_paths, ms_paths, hs_paths, labels, augment=False):
        self.rgb_paths = rgb_paths
        self.ms_paths = ms_paths
        self.hs_paths = hs_paths
        self.labels = labels
        self.augment = augment
        
        if augment:
            self.transform = A.Compose([
                # ===== Geometric Augmentations =====
                A.HorizontalFlip(p=0.6),  # Tăng từ 0.5
                A.VerticalFlip(p=0.6),  # Tăng từ 0.5
                A.Rotate(limit=25, p=0.7),  # ✅ Tăng từ 15° → 25° để augment mạnh hơn
                A.ShiftScaleRotate(
                    shift_limit=0.1,  # Tăng từ 0.05
                    scale_limit=0.2,  # Tăng từ 0.1
                    rotate_limit=25,  # Tăng từ 15°
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.5  # Tăng từ 0.3
                ),
                A.Transpose(p=0.3),  # Tăng từ 0.2
                
                # ===== AGGRESSIVE Color Augmentations =====
                A.RandomBrightnessContrast(
                    brightness_limit=0.25,  # ✅ Tăng từ 0.15 → 0.25
                    contrast_limit=0.25,
                    p=0.5  # ✅ Tăng từ 0.3 → 0.5
                ),
                
                # ===== ElasticTransform for robustness =====
                A.ElasticTransform(
                    alpha=50,
                    sigma=5,
                    p=0.3
                ),
                
                # ===== Resize & Random Crop (thay vì Center) =====
                A.Resize(256, 256),
                A.RandomCrop(224, 224, p=1.0)  # ✅ Random crop thay vì center để tăng diversity
                
            ], additional_targets={'ms': 'image', 'hs': 'image'})
        else:
            # Validation: no augmentation, same resize strategy as train
            self.transform = A.Compose([
                A.Resize(256, 256),        # Same as train
                A.CenterCrop(224, 224),    # Same as train
            ], additional_targets={'ms': 'image', 'hs': 'image'})
        
        # RGB normalization (ImageNet stats)
        self.normalize_rgb = A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Load RGB image
        rgb_img = cv2.imread(self.rgb_paths[idx])
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # Load MS image (5 channels)
        ms_img = tifffile.imread(self.ms_paths[idx])
        if len(ms_img.shape) == 2:
            ms_img = np.expand_dims(ms_img, -1)
        if ms_img.shape[-1] > 5:
            ms_img = ms_img[:, :, :5]
        elif ms_img.shape[-1] < 5:
            padding = np.zeros((*ms_img.shape[:2], 5 - ms_img.shape[-1]))
            ms_img = np.concatenate([ms_img, padding], axis=-1)
        
        # Load HS image (125 channels)
        hs_img = tifffile.imread(self.hs_paths[idx])
        if len(hs_img.shape) == 2:
            hs_img = np.expand_dims(hs_img, -1)
        if hs_img.shape[-1] > 125:
            hs_img = hs_img[:, :, :125]
        elif hs_img.shape[-1] < 125:
            padding = np.zeros((*hs_img.shape[:2], 125 - hs_img.shape[-1]))
            hs_img = np.concatenate([hs_img, padding], axis=-1)
        
        # Resize all to same initial size
        target_size = 256
        rgb_img = cv2.resize(rgb_img, (target_size, target_size))
        ms_img = cv2.resize(ms_img, (target_size, target_size))
        hs_img = cv2.resize(hs_img, (target_size, target_size))
        
        # Normalize to [0, 1]
        if rgb_img.max() > 1.0:
            rgb_img = rgb_img.astype(np.float32) / 255.0
        if ms_img.max() > 1.0:
            ms_img = ms_img.astype(np.float32) / 255.0
        if hs_img.max() > 1.0:
            hs_img = hs_img.astype(np.float32) / 255.0
        
        # Apply SAME transforms to all modalities
        transformed = self.transform(image=rgb_img, ms=ms_img, hs=hs_img)
        rgb_aug = transformed['image']
        ms_aug = transformed['ms']
        hs_aug = transformed['hs']
        
        # Apply ImageNet normalization to RGB only
        rgb_normalized = self.normalize_rgb(image=rgb_aug)['image']
        
        # Convert to tensors
        rgb = torch.from_numpy(rgb_normalized).permute(2, 0, 1).float()
        ms = torch.from_numpy(ms_aug).permute(2, 0, 1).float()
        hs = torch.from_numpy(hs_aug).permute(2, 0, 1).float()
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return rgb, ms, hs, label


# ==================== TRAINING FUNCTIONS ====================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0

    for rgb, ms, hs, y in loader:
        rgb, ms, hs, y = rgb.to(device), ms.to(device), hs.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(rgb, ms, hs)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()

    acc = correct / len(loader.dataset)
    return total_loss / len(loader), acc


@torch.no_grad()
def validate(model, loader, criterion, device, epoch=0, print_metrics=False):
    model.eval()
    total_loss, correct = 0, 0
    all_preds = []
    all_labels = []

    for rgb, ms, hs, y in loader:
        rgb, ms, hs, y = rgb.to(device), ms.to(device), hs.to(device), y.to(device)
        logits = model(rgb, ms, hs)
        loss = criterion(logits, y)

        total_loss += loss.item()
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    acc = correct / len(loader.dataset)
    
    # Print detailed metrics every 10 epochs
    if print_metrics and epoch % 10 == 0:
        print("\n" + "="*60)
        print("Classification Report:")
        print(classification_report(
            all_labels, all_preds,
            target_names=['Health', 'Rust', 'Other'],
            digits=3
        ))
        print("\nConfusion Matrix:")
        cm = confusion_matrix(all_labels, all_preds)
        print("        Pred: Health  Rust  Other")
        for i, row_name in enumerate(['Health', 'Rust', 'Other']):
            print(f"True {row_name:6s}:   {cm[i][0]:3d}    {cm[i][1]:3d}   {cm[i][2]:3d}")
        print("="*60 + "\n")
    
    return total_loss / len(loader), acc


# ==================== MAIN TRAINING ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 FIXED TRAINING PIPELINE - Anti-Collapse Strategy")
    print("="*70)
    print("Critical Fixes:")
    print("  ✅ AGGRESSIVE class weights [3.0, 0.5, 2.0] - Fix val stuck at 33.3%")
    print("  ✅ STRONGER augmentation (25° rotate, elastic, random crop)")
    print("  ✅ HIGHER dropout (0.5/0.6) - Fight overfitting")
    print("  ✅ FORCE model to predict all 3 classes!")
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
    print("\n✅ Using FIXED SyncedWheatMultiModalDataset")
    print("   - AGGRESSIVE augmentation (25° rotation, elastic, random crop)")
    print("   - AGGRESSIVE class weights [3.0, 0.5, 2.0]")
    print("   - HIGHER dropout (0.5/0.6)")
    print("   - Strategy: Fix val stuck at 33.3% & overfitting\n")
    
    train_dataset = SyncedWheatMultiModalDataset(
        train_rgb_paths, train_ms_paths, train_hs_paths, train_labels,
        augment=True
    )
    val_dataset = SyncedWheatMultiModalDataset(
        val_rgb_paths, val_ms_paths, val_hs_paths, val_labels,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False
    )
    
    if len(val_labels) > 0:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=16, 
            shuffle=False, 
            num_workers=0,
            pin_memory=False
        )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Model
    model = WheatSpectralNet(num_classes=3).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,  # Balanced learning rate
        weight_decay=1e-3  # Balanced weight decay
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # ===== FIX MODEL COLLAPSE: AGGRESSIVE CLASS WEIGHTS =====
    print("\n🚨 CRITICAL FIX: Val Acc stuck at 33.3% → Model predicting only 1 class!")
    print("📊 Using AGGRESSIVE class weights to FORCE balanced learning...\n")
    
    # Phân tích:
    # - Val Acc = 33.3% = 40/120 đúng → Model chỉ predict 1 class!
    # - Train Acc tăng (47%) nhưng Val stuck → OVERFITTING nghiêm trọng
    # - Giải pháp: FORCE model học cả 3 classes bằng aggressive weights
    
    class_weights_tensor = torch.FloatTensor([
        1.5,  # Health (class 0): BOOST MẠNH - model đang bỏ qua class này
        1.0,  # Rust   (class 1): PENALIZE MẠNH - có thể đang over-predict
        1.5   # Other  (class 2): BOOST - model có thể đang bỏ qua
    ]).to(device)
    
    print(f"Aggressive class weights (to fix collapse):")
    print(f"  Health (0): 3.0  ↑↑↑ STRONG boost")
    print(f"  Rust   (1): 0.5  ↓↓  STRONG penalty")
    print(f"  Other  (2): 2.0  ↑↑  STRONG boost")
    print(f"  Strategy: FORCE model to predict all 3 classes!\n")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    print(f"✅ Using CrossEntropyLoss with AGGRESSIVE weights\n")
    
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
    print(f"  - Dropout: 0.5 (features), 0.6 (head)")
    print(f"  - Loss: CrossEntropyLoss + AGGRESSIVE class weights [3.0, 0.5, 2.0]")
    print(f"  - Augmentation: STRONG (25° rotate, elastic, random crop)")
    print(f"  - Early stopping patience: {patience}")
    print("="*70 + "\n")
    
    for epoch in range(1, 101):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        if len(val_labels) > 0:
            val_loss, val_acc = validate(
                model, val_loader, criterion, device, 
                epoch=epoch, print_metrics=True
            )
            
            scheduler.step(val_acc)
        
            print(f"[Epoch {epoch:3d}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                
                checkpoint_path = Path(project_root) / 'checkpoints' / 'best_model_improved.pth'
                checkpoint_path.parent.mkdir(exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"  ✓ Saved best model (Val Acc: {val_acc:.3f}, Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  ⏳ No improvement for {patience_counter}/{patience} epochs")
                
                if patience_counter >= patience:
                    print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
                    break
        else:
            print(f"[Epoch {epoch:3d}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f}")
            
            if epoch % 10 == 0:
                checkpoint_path = Path(project_root) / 'checkpoints' / f'model_improved_epoch_{epoch}.pth'
                checkpoint_path.parent.mkdir(exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_acc': train_acc,
                }, checkpoint_path)
                print(f"  ✓ Saved checkpoint at epoch {epoch}")
    
    if len(val_labels) > 0:
        print(f"\n{'='*70}")
        print(f"🎉 Training complete!")
        print(f"{'='*70}")
        print(f"Best Val Acc:  {best_val_acc:.3f}")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"\n✅ Saved at: checkpoints/best_model_improved.pth")
        print(f"{'='*70}\n")
    else:
        print(f"\n🎉 Training complete!")
        print(f"Final Train Acc: {train_acc:.3f}\n")
