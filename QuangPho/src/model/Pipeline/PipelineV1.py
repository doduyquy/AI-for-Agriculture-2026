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
        self.context_gate = ContextGatedFusion(channels=512)

        # ===== Head =====
        self.head = RobustWheatHead(channels=512, num_classes=num_classes, dropout_rate=dropout_rate)

    def forward(self, rgb, ms, hs):
        """
        rgb: (B,3,H,W)
        ms : (B,5,H,W)
        hs : (B,125,H,W)
        """

        # --- ✅ RGB: Keep spatial feature map (7×7) ---
        F_rgb_map = self.rgb_backbone(rgb)  # (B, 512, 7, 7)
        B = F_rgb_map.size(0)
        # Dropout will be applied in head

        # --- MS ---
        F_ms_all = self.ms_extractor(ms)             # (B, 96)
        F_ms_all = self.dropout2(F_ms_all)  # Apply dropout
        F_ms_spec = F_ms_all[:, :64]
        F_ms_index = F_ms_all[:, 64:]

        # --- HS ---
        F_hs_all = self.hs_extractor(hs)
        F_hs_all = self.dropout2(F_hs_all)  # Apply dropout
        dim_sig = 375  # 125 * 3
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
        )                                           # (B,128)

        # --- ✅ Project fused vector to spatial map ---
        F_fused_proj_vec = self.fused_proj(F_fused_vec)  # (B, 512)
        F_spec_map = F_fused_proj_vec.view(B, 512, 1, 1)  # (B, 512, 1, 1)
        
        # Upsample spectral to match RGB spatial size (7×7)
        import torch.nn.functional as F
        F_spec_map_upsampled = F.interpolate(
            F_spec_map, 
            size=(7, 7), 
            mode='bilinear', 
            align_corners=False
        )  # (B, 512, 7, 7)

        # --- ✅ Gate với RGB: RGB (7×7) × Spectral (7×7) ---
        F_final = self.context_gate(F_rgb_map, F_spec_map_upsampled)  # (B, 512, 7, 7)

        # --- Classifier ---
        logits = self.head(F_final)
        return logits


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
    
    # Mapping labels
    label_map = {'Health': 0, 'Rust': 1, 'Other': 2}
    
    # Collect all files
    for label_name, label_id in label_map.items():
        # RGB files (.png)
        rgb_files = sorted(glob.glob(str(rgb_dir / f'{label_name}_hyper_*.png')))
        
        for rgb_file in rgb_files:
            # Extract ID from filename (e.g., Health_hyper_1.png -> 1)
            basename = Path(rgb_file).stem
            file_id = basename.split('_')[-1]
            
            # Corresponding MS and HS files (.tif)
            ms_file = ms_dir / f'{label_name}_hyper_{file_id}.tif'
            hs_file = hs_dir / f'{label_name}_hyper_{file_id}.tif'
            
            # Check if all files exist
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


class WheatMultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, rgb_paths, ms_data, hs_data, labels):
        self.rgb_paths = rgb_paths
        self.ms_data = ms_data
        self.hs_data = hs_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        rgb, _, _ = WheatRGBDataset(
            [self.rgb_paths[idx]],
            [self.labels[idx]]
        )[0]

        ms = torch.tensor(self.ms_data[idx], dtype=torch.float32)
        hs = torch.tensor(self.hs_data[idx], dtype=torch.float32)

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


if __name__ == "__main__":
    # Load data from folder structure
    data_root = Path(project_root) / 'Data'
    
    print("Loading training data...")
    train_rgb_paths, train_ms_paths, train_hs_paths, train_labels = load_data_from_folder(data_root, 'train')
    print(f"Train samples: {len(train_labels)}")
    print(f"  - Health: {train_labels.count(0)}")
    print(f"  - Rust: {train_labels.count(1)}")
    print(f"  - Other: {train_labels.count(2)}")
    
    print("\nLoading validation data...")
    val_rgb_paths, val_ms_paths, val_hs_paths, val_labels = load_data_from_folder(data_root, 'val')
    
    # If validation folder is empty or has very few samples, split from training data
    if len(val_labels) < 50:  # Threshold for valid validation set
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
    
    # Create datasets with SYNCED augmentation (fixes data leakage)
    print("\n✅ Using SyncedWheatMultiModalDataset with Albumentations")
    print("   - Geometric transforms applied to RGB, MS, HS simultaneously")
    print("   - No spatial mismatch between modalities\n")
    
    train_dataset = SyncedWheatMultiModalDataset(
        train_rgb_paths, train_ms_paths, train_hs_paths, train_labels,
        augment=True  # Enable synced augmentation for training
    )
    val_dataset = SyncedWheatMultiModalDataset(
        val_rgb_paths, val_ms_paths, val_hs_paths, val_labels,
        augment=False  # No augmentation for validation
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False  # Disable pin_memory on CPU
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
    print(f"\nUsing device: {device}\n")
    
    model = WheatSpectralNet(num_classes=3).to(device)
    
    # Optimizer with lower LR and higher weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,  # Giảm từ 3e-4 xuống 1e-4
        weight_decay=1e-3  # Tăng từ 1e-4 lên 1e-3
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop with early stopping
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    print("Starting training with improvements...")
    print(f"- Data augmentation: enabled")
    print(f"- Dropout: 0.3-0.5")
    print(f"- Learning rate: 1e-4")
    print(f"- Weight decay: 1e-3")
    print(f"- Early stopping patience: {patience}\n")
    
    for epoch in range(1, 101):  # Tăng max epochs
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Only validate if we have validation data
        if len(val_labels) > 0:
            val_loss, val_acc = validate(
                model, val_loader, criterion, device, 
                epoch=epoch, print_metrics=True
            )
            
            # Update learning rate
            scheduler.step(val_acc)
        
            print(f"[Epoch {epoch:3d}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")
            
            # Save best model based on validation
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                
                checkpoint_path = Path(project_root) / 'checkpoints' / 'best_model.pth'
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
            # No validation - just train
            print(f"[Epoch {epoch:3d}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f}")
            
            # Save model every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = Path(project_root) / 'checkpoints' / f'model_epoch_{epoch}.pth'
                checkpoint_path.parent.mkdir(exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_acc': train_acc,
                }, checkpoint_path)
                print(f"  ✓ Saved checkpoint at epoch {epoch}")
    
    if len(val_labels) > 0:
        print(f"\n🎉 Training complete!")
        print(f"Best Val Acc: {best_val_acc:.3f}")
        print(f"Best Val Loss: {best_val_loss:.4f}")
    else:
        print(f"\n🎉 Training complete!")
        print(f"Final Train Acc: {train_acc:.3f}")
