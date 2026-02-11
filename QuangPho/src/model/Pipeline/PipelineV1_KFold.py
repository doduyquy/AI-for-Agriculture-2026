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
    # Fallback: use torchvision (với cảnh báo)
    print("⚠️  WARNING: Falling back to torchvision transforms")
    print("⚠️  This may cause data leakage (augmentation not synced)")
    print("⚠️  Please install albumentations for best results!\n")
    A = None

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

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

        # ===== Unified Dropout for regularization =====
        dropout_rate = 0.3  # Consistent dropout rate
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
        self.context_gate = ContextGatedFusion(
            spatial_channels=512,   # RGB map channels
            context_dim=512,        # Fused spectral vector projected
            out_channels=512
        )

        # ===== Final Classification Head =====
        self.head = RobustWheatHead(
            in_features=512,
            num_classes=num_classes,
            dropout=dropout_rate
        )

    def forward(self, rgb_img, ms_img, hs_img):
        # RGB feature map (B, 512, 7, 7)
        F_rgb_map = self.rgb_backbone(rgb_img)

        # MS features (B, 96)
        F_ms = self.ms_extractor(ms_img)
        F_ms = self.dropout1(F_ms)

        # HS features (B, 507)
        F_hs = self.hs_extractor(hs_img)
        F_hs = self.dropout2(F_hs)

        # Fusion (B, 128)
        F_fused_vec = self.fusion(F_ms, F_hs)

        # Project to 512-dim vector
        F_fused_vec_512 = self.fused_proj(F_fused_vec)  # (B, 512)

        # Upsample spectral feature to match RGB spatial size (7x7)
        B = F_fused_vec_512.size(0)
        F_fused_map = F_fused_vec_512.view(B, 512, 1, 1).expand(B, 512, 7, 7)

        # Context Gate: Combine RGB spatial map with spectral features
        F_gated = self.context_gate(F_rgb_map, F_fused_map)  # (B, 512, 7, 7)

        # Classification
        logits = self.head(F_gated)

        return logits


def load_data_from_folder(data_root, split_name):
    """
    Load RGB, MS, HS image paths and labels from Data/train or Data/val folder
    
    Expected structure:
        Data/
            train/
                RGB/
                    Health_hyper_001.png
                    Rust_hyper_001.png
                    Other_hyper_001.png
                MS/
                    Health_hyper_001.tif
                    Rust_hyper_001.tif
                    Other_hyper_001.tif
                HS/
                    Health_hyper_001.tif
                    Rust_hyper_001.tif
                    Other_hyper_001.tif
    """
    rgb_folder = data_root / split_name / 'RGB'
    ms_folder = data_root / split_name / 'MS'
    hs_folder = data_root / split_name / 'HS'
    
    # Label mapping
    label_map = {
        'Health': 0,
        'Rust': 1,
        'Other': 2
    }
    
    rgb_paths = []
    ms_paths = []
    hs_paths = []
    labels = []
    
    # Get all RGB images
    if not rgb_folder.exists():
        print(f"Warning: {rgb_folder} does not exist")
        return [], [], [], []
    
    for label_name, label_id in label_map.items():
        # Find all files starting with label_name
        pattern = str(rgb_folder / f"{label_name}_hyper_*.png")
        rgb_files = sorted(glob.glob(pattern))
        
        for rgb_file in rgb_files:
            # Extract filename without extension
            basename = Path(rgb_file).stem  # e.g., "Health_hyper_001"
            
            # Find corresponding MS and HS files
            ms_file = ms_folder / f"{basename}.tif"
            hs_file = hs_folder / f"{basename}.tif"
            
            # Only add if all three modalities exist
            if ms_file.exists() and hs_file.exists():
                rgb_paths.append(str(rgb_file))
                ms_paths.append(str(ms_file))
                hs_paths.append(str(hs_file))
                labels.append(label_id)
    
    return rgb_paths, ms_paths, hs_paths, labels


class SyncedWheatMultiModalDataset(torch.utils.data.Dataset):
    """
    Dataset with SYNCHRONIZED augmentation for RGB, MS, HS using Albumentations
    Fixes data leakage issue by applying same geometric transforms to all modalities
    """
    def __init__(self, rgb_paths, ms_paths, hs_paths, labels, augment=False):
        self.rgb_paths = rgb_paths
        self.ms_paths = ms_paths
        self.hs_paths = hs_paths
        self.labels = labels
        self.augment = augment
        
        # Albumentations: Apply SAME transform to all modalities
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),  # Giảm từ 15 xuống 10
                A.RandomBrightnessContrast(
                    p=0.2, 
                    brightness_limit=0.1, 
                    contrast_limit=0.1
                ),
                A.Resize(224, 224),
                A.CenterCrop(224, 224),
            ], additional_targets={'ms': 'image', 'hs': 'image'})
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.CenterCrop(224, 224),
            ], additional_targets={'ms': 'image', 'hs': 'image'})
        
        # RGB normalization (ImageNet stats)
        self.normalize_rgb = A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Load RGB image (H, W, 3)
        rgb_img = cv2.imread(self.rgb_paths[idx])
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # Load MS image (H, W, 5) using tifffile
        ms_img = tifffile.imread(self.ms_paths[idx])
        if len(ms_img.shape) == 2:
            ms_img = np.expand_dims(ms_img, -1)
        
        # Ensure exactly 5 channels
        if ms_img.shape[-1] > 5:
            ms_img = ms_img[:, :, :5]
        elif ms_img.shape[-1] < 5:
            padding = np.zeros((*ms_img.shape[:2], 5 - ms_img.shape[-1]))
            ms_img = np.concatenate([ms_img, padding], axis=-1)
        
        # Load HS image (H, W, 125) using tifffile
        hs_img = tifffile.imread(self.hs_paths[idx])
        if len(hs_img.shape) == 2:
            hs_img = np.expand_dims(hs_img, -1)
        
        # Ensure exactly 125 channels
        if hs_img.shape[-1] > 125:
            hs_img = hs_img[:, :, :125]
        elif hs_img.shape[-1] < 125:
            padding = np.zeros((*hs_img.shape[:2], 125 - hs_img.shape[-1]))
            hs_img = np.concatenate([hs_img, padding], axis=-1)
        
        # ✅ Resize all images to same size FIRST (before Albumentations)
        # This is required because RGB, MS, HS may have different dimensions
        target_size = 256  # Resize to slightly larger, then crop to 224
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
        
        # ✅ Apply SAME geometric transforms to all modalities
        transformed = self.transform(image=rgb_img, ms=ms_img, hs=hs_img)
        rgb_aug = transformed['image']
        ms_aug = transformed['ms']
        hs_aug = transformed['hs']
        
        # Apply ImageNet normalization ONLY to RGB
        rgb_normalized = self.normalize_rgb(image=rgb_aug)['image']
        
        # Convert to tensors (H, W, C) -> (C, H, W)
        rgb = torch.from_numpy(rgb_normalized).permute(2, 0, 1).float()
        ms = torch.from_numpy(ms_aug).permute(2, 0, 1).float()
        hs = torch.from_numpy(hs_aug).permute(2, 0, 1).float()
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return rgb, ms, hs, label


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


def train_with_kfold(k_folds=5):
    """
    Train model with K-Fold Cross Validation
    """
    # Load ALL data from train folder
    data_root = Path(project_root) / 'Data'
    all_rgb_paths, all_ms_paths, all_hs_paths, all_labels = load_data_from_folder(data_root, 'train')
    
    print(f"Total samples: {len(all_labels)}")
    print(f"  - Health: {all_labels.count(0)}")
    print(f"  - Rust: {all_labels.count(1)}")
    print(f"  - Other: {all_labels.count(2)}")
    print(f"\n{'='*70}")
    print(f"Starting {k_folds}-Fold Cross Validation")
    print(f"{'='*70}\n")
    
    # Setup K-Fold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Store results
    fold_results = []
    
    # K-Fold loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_rgb_paths, all_labels), 1):
        print(f"\n{'='*70}")
        print(f"FOLD {fold}/{k_folds}")
        print(f"{'='*70}")
        
        # Split data for this fold
        train_rgb = [all_rgb_paths[i] for i in train_idx]
        train_ms = [all_ms_paths[i] for i in train_idx]
        train_hs = [all_hs_paths[i] for i in train_idx]
        train_y = [all_labels[i] for i in train_idx]
        
        val_rgb = [all_rgb_paths[i] for i in val_idx]
        val_ms = [all_ms_paths[i] for i in val_idx]
        val_hs = [all_hs_paths[i] for i in val_idx]
        val_y = [all_labels[i] for i in val_idx]
        
        print(f"Train samples: {len(train_y)} | Val samples: {len(val_y)}")
        print(f"Train - Health: {train_y.count(0)}, Rust: {train_y.count(1)}, Other: {train_y.count(2)}")
        print(f"Val   - Health: {val_y.count(0)}, Rust: {val_y.count(1)}, Other: {val_y.count(2)}\n")
        
        # Create datasets
        train_dataset = SyncedWheatMultiModalDataset(
            train_rgb, train_ms, train_hs, train_y, augment=True
        )
        val_dataset = SyncedWheatMultiModalDataset(
            val_rgb, val_ms, val_hs, val_y, augment=False
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True, 
            num_workers=0, pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False,
            num_workers=0, pin_memory=False
        )
        
        # Initialize model for this fold
        model = WheatSpectralNet(num_classes=3).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=1e-3
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training loop for this fold
        best_val_acc = 0.0
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(1, 51):  # 50 epochs per fold
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc = validate(
                model, val_loader, criterion, device, 
                epoch=epoch, print_metrics=(epoch % 10 == 0)
            )
            
            scheduler.step(val_acc)
            
            if epoch % 5 == 0:  # Print every 5 epochs
                print(f"  [Epoch {epoch:2d}] "
                      f"Train: Loss={train_loss:.4f}, Acc={train_acc:.3f} | "
                      f"Val: Loss={val_loss:.4f}, Acc={val_acc:.3f}")
            
            # Track best for this fold
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model for this fold
                checkpoint_path = Path(project_root) / 'checkpoints' / f'fold_{fold}_best.pth'
                checkpoint_path.parent.mkdir(exist_ok=True)
                torch.save({
                    'fold': fold,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  ⚠️ Early stopping at epoch {epoch}")
                    break
        
        # Store fold result
        fold_results.append({
            'fold': fold,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss
        })
        
        print(f"\n✓ Fold {fold} completed:")
        print(f"  Best Val Acc:  {best_val_acc:.4f}")
        print(f"  Best Val Loss: {best_val_loss:.4f}")
    
    # ===== Summary =====
    print(f"\n{'='*70}")
    print(f"K-FOLD CROSS VALIDATION RESULTS")
    print(f"{'='*70}")
    
    accs = [r['best_val_acc'] for r in fold_results]
    losses = [r['best_val_loss'] for r in fold_results]
    
    print("\nPer-fold results:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: Val Acc = {r['best_val_acc']:.4f}, Val Loss = {r['best_val_loss']:.4f}")
    
    print(f"\n{'='*70}")
    print(f"AVERAGE RESULTS ({k_folds}-Fold CV):")
    print(f"  Mean Val Acc:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Mean Val Loss: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    print(f"  Min Val Acc:   {np.min(accs):.4f}")
    print(f"  Max Val Acc:   {np.max(accs):.4f}")
    print(f"{'='*70}\n")
    
    return fold_results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔬 K-FOLD CROSS VALIDATION TRAINING")
    print("="*70 + "\n")
    
    # Run K-Fold Cross Validation
    results = train_with_kfold(k_folds=5)
    
    print("\n🎉 K-Fold Cross Validation Complete!")
    print(f"📊 Results saved in: {Path(project_root) / 'checkpoints'}")
    print("\nCheckpoint files:")
    for i in range(1, 6):
        print(f"  - fold_{i}_best.pth")
