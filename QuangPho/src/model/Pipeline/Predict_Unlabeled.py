"""
Predict_Unlabeled.py - DỰ ĐOÁN DATA KHÔNG CÓ NHÃN

🎯 Mục đích:
  - Dự đoán các file trong Data/val/ KHÔNG CÓ NHÃN
  - Export kết quả ra file CSV với: Image_Path, Predicted_Label, Predicted_Class

🔧 Quy trình:
  1. Load models (Step 1 và Step 2)
  2. Load tất cả ảnh trong val/RGB (không cần nhãn)
  3. Chạy two-step inference
  4. Export CSV: val_predictions.csv
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import tifffile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from torch.utils.data import Dataset, DataLoader
from src.model.Layer1.CNNRGB import RGBCNNFeature
from src.model.Layer2.AttentionFusion import MultiHeadSpectralAttention
from src.FeatureEngineering.RGBFeature import RGBFeature


# ==================== MODEL ARCHITECTURE ====================
class AsymmetricBiasedGate(nn.Module):
    """Biased gate handling ASYMMETRIC dimensions"""
    def __init__(self, rgb_channels=512, spec_channels=256, rgb_weight=0.65):
        super().__init__()
        self.rgb_channels = rgb_channels
        self.spec_channels = spec_channels
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


class TwoStepWheatNet(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super().__init__()
        
        # RGB backbone (frozen)
        self.rgb_backbone = RGBCNNFeature(backbone='resnet18', pretrained=True)
        for param in self.rgb_backbone.parameters():
            param.requires_grad = False
        
        # MS encoder
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
        
        # HS encoder
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
        
        # Spectral attention fusion
        self.spectral_attention = MultiHeadSpectralAttention(
            ms_channels=128,
            hs_channels=128,
            num_heads=4,
            dropout=dropout_rate
        )
        
        # Spectral projection (keep 256 channels)
        self.spectral_proj = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        
        # RGB handcrafted features
        self.rgb_handcrafted_proj = nn.Sequential(
            nn.Linear(84, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128)
        )
        
        self.rgb_fusion = nn.Linear(640, 512)
        
        # Step 1: Healthy vs Diseased
        self.step1_gate = AsymmetricBiasedGate(
            rgb_channels=512, 
            spec_channels=256,
            rgb_weight=0.65
        )
        
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
        
        # Step 2: Rust vs Other
        self.step2_gate = AsymmetricBiasedGate(
            rgb_channels=512,
            spec_channels=256,
            rgb_weight=0.30
        )
        
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
    
    def forward(self, rgb, ms, hs, rgb_handcrafted=None, step='both'):
        # Extract shared features
        with torch.no_grad():
            f_rgb_cnn = self.rgb_backbone(rgb)
        
        f_ms = self.ms_encoder(ms)
        f_hs = self.hs_encoder(hs)
        f_spec = self.spectral_attention(f_ms, f_hs)
        f_spec = self.spectral_proj(f_spec)
        
        # Enhance RGB with handcrafted features
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
        
        else:  # step == 'both'
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
            final_logits = torch.log(final_probs + 1e-8)
            
            return final_logits


# ==================== DATASET FOR UNLABELED DATA ====================
class UnlabeledDataset(Dataset):
    """Dataset for unlabeled data (no labels in filename)"""
    def __init__(self, rgb_paths, ms_paths, hs_paths):
        self.rgb_paths = rgb_paths
        self.ms_paths = ms_paths
        self.hs_paths = hs_paths
        
        # Import albumentations
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        # RGB transform (no augmentation for inference)
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
        return len(self.rgb_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        rgb_path = self.rgb_paths[idx]
        rgb = np.array(Image.open(rgb_path).convert('RGB')).astype(np.float32) / 255.0
        ms = tifffile.imread(self.ms_paths[idx]).astype(np.float32)
        hs = tifffile.imread(self.hs_paths[idx]).astype(np.float32)
        
        # Extract handcrafted features
        try:
            img_orig = cv2.imread(rgb_path)
            img_orig_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
            
            glcm_feat, _ = RGBFeature.glcm_rgb(rgb_path)
            lbp_feat = RGBFeature.lbp_multiscale(img_orig_rgb[:, :, 1])
            color_feat = RGBFeature.rgb_color_veg_features(rgb_path)
            
            handcrafted = np.concatenate([glcm_feat, lbp_feat, color_feat])
            handcrafted = torch.from_numpy(handcrafted).float()
        except Exception as e:
            print(f"Warning: Failed to extract handcrafted features for {rgb_path}: {e}")
            handcrafted = torch.zeros(84)
        
        # Apply transforms
        rgb_transformed = self.rgb_transform(image=rgb)['image']
        ms_transformed = self.ms_transform(image=ms)['image']
        hs_transformed = self.hs_transform(image=hs)['image']
        
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
        
        ms_tensor = torch.clamp(ms_tensor, 0, 1)
        hs_tensor = torch.clamp(hs_tensor, 0, 1)
        
        return rgb_transformed, ms_tensor, hs_tensor, handcrafted


# ==================== LOAD UNLABELED DATA ====================
def load_unlabeled_data(data_root):
    """Load all images from val folder (no labels needed)"""
    rgb_folder = data_root / 'val' / 'RGB'
    ms_folder = data_root / 'val' / 'MS'
    hs_folder = data_root / 'val' / 'HS'
    
    rgb_paths, ms_paths, hs_paths = [], [], []
    
    # Get all RGB files (regardless of naming pattern)
    rgb_files = sorted(list(rgb_folder.glob('*.png')) + list(rgb_folder.glob('*.tif')))
    
    for rgb_file in rgb_files:
        stem = rgb_file.stem
        
        # Find corresponding MS and HS files
        ms_file = ms_folder / f"{stem}.tif"
        hs_file = hs_folder / f"{stem}.tif"
        
        if ms_file.exists() and hs_file.exists():
            rgb_paths.append(str(rgb_file))
            ms_paths.append(str(ms_file))
            hs_paths.append(str(hs_file))
    
    return rgb_paths, ms_paths, hs_paths


# ==================== TWO-STEP PREDICTION ====================
@torch.no_grad()
def predict_unlabeled(model_step1, model_step2, loader, device, dataset):
    """
    Predict unlabeled data using two-step approach
    
    Returns:
        predictions, rgb_paths
    """
    model_step1.eval()
    model_step2.eval()
    
    all_preds = []
    all_indices = []
    
    print("\n🔍 Running Two-Step Prediction on Unlabeled Data...")
    print("="*70)
    
    step1_healthy = 0
    step1_diseased = 0
    step2_rust = 0
    step2_other = 0
    
    for batch_idx, (rgb, ms, hs, handcrafted) in enumerate(loader):
        rgb = rgb.to(device)
        ms = ms.to(device)
        hs = hs.to(device)
        handcrafted = handcrafted.to(device)
        batch_size = rgb.size(0)
        
        # Track indices
        batch_start_idx = batch_idx * loader.batch_size
        batch_indices = list(range(batch_start_idx, batch_start_idx + batch_size))
        
        final_predictions = torch.zeros(batch_size, dtype=torch.long)
        
        # Step 1: Healthy vs Diseased
        step1_outputs = model_step1(rgb, ms, hs, rgb_handcrafted=handcrafted, step='step1')
        step1_preds = step1_outputs.argmax(dim=1)
        
        step1_healthy += (step1_preds == 0).sum().item()
        step1_diseased += (step1_preds == 1).sum().item()
        
        # Healthy samples
        healthy_mask = step1_preds == 0
        final_predictions[healthy_mask] = 0
        
        # Diseased samples → run Step 2
        diseased_mask = step1_preds == 1
        if diseased_mask.sum() > 0:
            rgb_diseased = rgb[diseased_mask]
            ms_diseased = ms[diseased_mask]
            hs_diseased = hs[diseased_mask]
            
            step2_outputs = model_step2(rgb_diseased, ms_diseased, hs_diseased, step='step2')
            step2_preds = step2_outputs.argmax(dim=1)
            
            step2_rust += (step2_preds == 0).sum().item()
            step2_other += (step2_preds == 1).sum().item()
            
            # Map: 0=Rust→1, 1=Other→2
            step2_final = step2_preds + 1
            final_predictions[diseased_mask] = step2_final
        
        all_preds.extend(final_predictions.cpu().numpy())
        all_indices.extend(batch_indices)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {(batch_idx + 1) * batch_size} samples...")
    
    print("="*70)
    print(f"\n📊 Step 1 Predictions:")
    print(f"  Healthy: {step1_healthy}")
    print(f"  Diseased: {step1_diseased}")
    print(f"\n📊 Step 2 Predictions (on diseased samples):")
    print(f"  Rust: {step2_rust}")
    print(f"  Other: {step2_other}")
    print(f"\n📊 Final 3-Class Predictions:")
    print(f"  Health: {step1_healthy}")
    print(f"  Rust: {step2_rust}")
    print(f"  Other: {step2_other}")
    print("="*70 + "\n")
    
    # Get RGB paths
    rgb_paths = [dataset.rgb_paths[i] for i in all_indices]
    
    return np.array(all_preds), rgb_paths


# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 PREDICT UNLABELED DATA - VAL SET")
    print("="*70)
    print("Models:")
    print("  Step 1: checkpoints/best_model_step1.pth")
    print("  Step 2: checkpoints/best_model_step2.pth")
    print("Data:")
    print("  Source: Data/val/")
    print("  Output: val_predictions.csv")
    print("="*70 + "\n")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    data_root = project_root / 'Data'
    print("📂 Loading unlabeled data from val folder...")
    rgb_paths, ms_paths, hs_paths = load_unlabeled_data(data_root)
    print(f"✅ Found {len(rgb_paths)} samples\n")
    
    if len(rgb_paths) == 0:
        print("❌ No data found in val folder!")
        sys.exit(1)
    
    # Create dataset and loader
    dataset = UnlabeledDataset(rgb_paths, ms_paths, hs_paths)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Load models
    print("🏗️ Building models...")
    model_step1 = TwoStepWheatNet(dropout_rate=0.4).to(device)
    model_step2 = TwoStepWheatNet(dropout_rate=0.4).to(device)
    print("✅ Models built\n")
    
    # Load checkpoints
    checkpoint_step1 = project_root / 'checkpoints' / 'best_model_step1.pth'
    checkpoint_step2 = project_root / 'checkpoints' / 'best_model_step2.pth'
    
    print(f"📥 Loading Step 1 checkpoint: {checkpoint_step1}")
    checkpoint_data = torch.load(checkpoint_step1, map_location=device)
    if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
        model_step1.load_state_dict(checkpoint_data['model_state_dict'])
        print(f"  ✅ Loaded from epoch {checkpoint_data.get('epoch', 'unknown')}")
    else:
        model_step1.load_state_dict(checkpoint_data)
        print(f"  ✅ Loaded state_dict")
    
    print(f"📥 Loading Step 2 checkpoint: {checkpoint_step2}")
    checkpoint_data = torch.load(checkpoint_step2, map_location=device)
    if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
        model_step2.load_state_dict(checkpoint_data['model_state_dict'])
        print(f"  ✅ Loaded from epoch {checkpoint_data.get('epoch', 'unknown')}")
    else:
        model_step2.load_state_dict(checkpoint_data)
        print(f"  ✅ Loaded state_dict")
    
    print("✅ Models loaded successfully!\n")
    
    # Run prediction
    predictions, image_paths = predict_unlabeled(
        model_step1, model_step2, loader, device, dataset
    )
    
    # Save to CSV
    import csv
    output_csv = project_root / 'val_predictions.csv'
    
    label_names = {0: 'Health', 1: 'Rust', 2: 'Other'}
    
    print("\n💾 Saving predictions to CSV...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header (only id and Predicted_Class)
        writer.writerow(['id', 'Predicted_Class'])
        
        # Write predictions
        for img_path, pred_label in zip(image_paths, predictions):
            filename = Path(img_path).name
            pred_class = label_names[pred_label]
            
            writer.writerow([
                filename,
                pred_class
            ])
    
    print(f"✅ Predictions saved to: {output_csv}")
    print(f"📊 Total samples: {len(predictions)}")
    print(f"📊 Distribution:")
    print(f"  Health: {(predictions == 0).sum()}")
    print(f"  Rust: {(predictions == 1).sum()}")
    print(f"  Other: {(predictions == 2).sum()}")
    print("\n" + "="*70)
    print("✅ PREDICTION COMPLETED!")
    print("="*70 + "\n")
