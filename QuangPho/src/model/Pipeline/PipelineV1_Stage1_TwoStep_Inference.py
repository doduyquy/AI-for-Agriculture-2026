"""
PipelineV1_Stage1_TwoStep_Inference.py - FINAL 3-CLASS INFERENCE

🎯 Mục đích:
  - Load riêng biệt Step 1 model (best_model_step1.pth)
  - Load riêng biệt Step 2 model (best_model_step2.pth)
  - Kết hợp 2 steps để predict 3 classes: Health, Rust, Other

🔧 Quy trình:
  1. Step 1: Predict Healthy vs Diseased
  2. If Healthy → Final prediction = 0 (Health)
  3. If Diseased → Step 2: Predict Rust vs Other
     - If Rust → Final prediction = 1 (Rust)
     - If Other → Final prediction = 2 (Other)

📊 Output:
  - Final 3-class classification report
  - Confusion matrix
  - Per-class accuracy
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import cv2

# Install dependencies
try:
    import tifffile
except ImportError:
    import subprocess
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
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image

from sklearn.metrics import classification_report, confusion_matrix

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader
from src.model.Layer1.CNNRGB import RGBCNNFeature
from src.model.Layer2.AttentionFusion import MultiHeadSpectralAttention
from src.FeatureEngineering.RGBFeature import RGBFeature


# ==================== MODEL ARCHITECTURE ====================
# (Copy từ PipelineV1_Stage1_TwoStep.py)

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


# ==================== DATASET ====================
class Stage1Dataset(torch.utils.data.Dataset):
    """Dataset for inference"""
    def __init__(self, rgb_paths, ms_paths, hs_paths, labels, use_handcrafted=True):
        self.rgb_paths = rgb_paths
        self.ms_paths = ms_paths
        self.hs_paths = hs_paths
        self.labels = labels
        self.use_handcrafted = use_handcrafted
        
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
        return len(self.labels)
    
    def __getitem__(self, idx):
        rgb_path = self.rgb_paths[idx]
        rgb = np.array(Image.open(rgb_path).convert('RGB')).astype(np.float32) / 255.0
        ms = tifffile.imread(self.ms_paths[idx]).astype(np.float32)
        hs = tifffile.imread(self.hs_paths[idx]).astype(np.float32)
        label = self.labels[idx]
        
        # Extract handcrafted features
        if self.use_handcrafted:
            try:
                img_orig = cv2.imread(rgb_path)
                img_orig_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
                
                glcm_feat, _ = RGBFeature.glcm_rgb(rgb_path)
                lbp_feat = RGBFeature.lbp_multiscale(img_orig_rgb[:, :, 1])
                color_feat = RGBFeature.rgb_color_veg_features(rgb_path)
                
                handcrafted = np.concatenate([glcm_feat, lbp_feat, color_feat])
                handcrafted = torch.from_numpy(handcrafted).float()
            except Exception as e:
                handcrafted = torch.zeros(84)
        else:
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
        
        return rgb_transformed, ms_tensor, hs_tensor, handcrafted, label


# ==================== DATA LOADING ====================
def load_data_from_folder(data_root, split='val'):
    """Load RGB, MS, HS paths and labels"""
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


# ==================== TWO-STEP INFERENCE ====================
@torch.no_grad()
def two_step_inference(model_step1, model_step2, loader, device, dataset):
    """
    Two-step inference:
    1. Run Step 1 (Healthy vs Diseased)
    2. For Diseased samples, run Step 2 (Rust vs Other)
    3. Combine results to get final 3-class predictions
    
    Args:
        dataset: Dataset object to get image paths
    
    Returns:
        predictions, true_labels, rgb_paths
    """
    model_step1.eval()
    model_step2.eval()
    
    all_preds = []
    all_labels = []
    all_indices = []
    
    print("\n🔍 Running Two-Step Inference...")
    print("="*70)
    
    step1_healthy_count = 0
    step1_diseased_count = 0
    step2_rust_count = 0
    step2_other_count = 0
    
    for batch_idx, (rgb, ms, hs, handcrafted, labels) in enumerate(loader):
        rgb = rgb.to(device)
        ms = ms.to(device)
        hs = hs.to(device)
        handcrafted = handcrafted.to(device)
        batch_size = rgb.size(0)
        
        # Track indices for this batch
        batch_start_idx = batch_idx * loader.batch_size
        batch_indices = list(range(batch_start_idx, batch_start_idx + batch_size))
        
        final_predictions = torch.zeros(batch_size, dtype=torch.long)
        
        # Step 1: Healthy vs Diseased
        step1_outputs = model_step1(rgb, ms, hs, rgb_handcrafted=handcrafted, step='step1')
        step1_preds = step1_outputs.argmax(dim=1)  # 0=Healthy, 1=Diseased
        
        # Count Step 1 predictions
        step1_healthy_count += (step1_preds == 0).sum().item()
        step1_diseased_count += (step1_preds == 1).sum().item()
        
        # For samples predicted as Healthy, final prediction = 0
        healthy_mask = step1_preds == 0
        final_predictions[healthy_mask] = 0
        
        # For samples predicted as Diseased, run Step 2
        diseased_mask = step1_preds == 1
        if diseased_mask.sum() > 0:
            rgb_diseased = rgb[diseased_mask]
            ms_diseased = ms[diseased_mask]
            hs_diseased = hs[diseased_mask]
            
            # Step 2: Rust vs Other (on diseased samples only)
            step2_outputs = model_step2(rgb_diseased, ms_diseased, hs_diseased, step='step2')
            step2_preds = step2_outputs.argmax(dim=1)  # 0=Rust, 1=Other
            
            # Count Step 2 predictions
            step2_rust_count += (step2_preds == 0).sum().item()
            step2_other_count += (step2_preds == 1).sum().item()
            
            # Map Step 2 predictions to final 3-class
            # 0=Rust → 1, 1=Other → 2
            step2_final = step2_preds + 1
            final_predictions[diseased_mask] = step2_final
        
        all_preds.extend(final_predictions.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_indices.extend(batch_indices)
        
        if (batch_idx + 1) % 5 == 0:
            print(f"  Processed {(batch_idx + 1) * batch_size} samples...")
    
    # Get RGB paths from dataset using indices
    rgb_paths = [dataset.rgb_paths[i] for i in all_indices]
    
    print("="*70)
    print(f"\n📊 Step 1 Predictions:")
    print(f"  - Healthy: {step1_healthy_count}")
    print(f"  - Diseased: {step1_diseased_count}")
    print(f"\n📊 Step 2 Predictions (on diseased samples):")
    print(f"  - Rust: {step2_rust_count}")
    print(f"  - Other: {step2_other_count}")
    print(f"\n📊 Final 3-Class Predictions:")
    print(f"  - Health: {step1_healthy_count}")
    print(f"  - Rust: {step2_rust_count}")
    print(f"  - Other: {step2_other_count}")
    print("="*70 + "\n")
    
    return np.array(all_preds), np.array(all_labels), rgb_paths


# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 TWO-STEP INFERENCE - FINAL 3-CLASS PREDICTION")
    print("="*70)
    print("Loading:")
    print("  1. Step 1 model: checkpoints/best_model_step1.pth")
    print("  2. Step 2 model: checkpoints/best_model_step2.pth")
    print("="*70 + "\n")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Data paths
    data_root = Path(__file__).parent.parent.parent.parent / 'Data'
    
    # Load validation data
    print("📂 Loading validation data...")
    val_rgb_paths, val_ms_paths, val_hs_paths, val_labels = load_data_from_folder(data_root, 'val')
    
    if len(val_labels) == 0:
        print("⚠️ No validation data found. Using train data split...")
        train_rgb_paths, train_ms_paths, train_hs_paths, train_labels = load_data_from_folder(data_root, 'train')
        
        from sklearn.model_selection import train_test_split
        _, val_rgb_paths, _, val_ms_paths, _, val_hs_paths, _, val_labels = train_test_split(
            train_rgb_paths, train_ms_paths, train_hs_paths, train_labels,
            test_size=0.2, random_state=42, stratify=train_labels
        )
    
    print(f"✅ Loaded {len(val_labels)} validation samples")
    print(f"  - Health: {val_labels.count(0)}")
    print(f"  - Rust: {val_labels.count(1)}")
    print(f"  - Other: {val_labels.count(2)}\n")
    
    # Create dataset and dataloader
    val_dataset = Stage1Dataset(val_rgb_paths, val_ms_paths, val_hs_paths, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Create models
    print("🏗️ Building models...")
    model_step1 = TwoStepWheatNet(dropout_rate=0.4).to(device)
    model_step2 = TwoStepWheatNet(dropout_rate=0.4).to(device)
    
    # Load checkpoints
    checkpoint_step1 = Path('checkpoints/best_model_step1.pth')
    checkpoint_step2 = Path('checkpoints/best_model_step2.pth')
    
    if not checkpoint_step1.exists():
        print(f"❌ Error: {checkpoint_step1} not found!")
        print("Please train the model first using PipelineV1_Stage1_TwoStep.py")
        sys.exit(1)
    
    if not checkpoint_step2.exists():
        print(f"❌ Error: {checkpoint_step2} not found!")
        print("Please train the model first using PipelineV1_Stage1_TwoStep.py")
        sys.exit(1)
    
    # Load Step 1 checkpoint with format checking
    print(f"📥 Loading Step 1 checkpoint: {checkpoint_step1}")
    checkpoint_data_step1 = torch.load(checkpoint_step1, map_location=device)
    
    if isinstance(checkpoint_data_step1, dict) and 'model_state_dict' in checkpoint_data_step1:
        # Full checkpoint with metadata
        model_step1.load_state_dict(checkpoint_data_step1['model_state_dict'])
        print(f"  ✅ Loaded from epoch {checkpoint_data_step1.get('epoch', 'unknown')}")
        if 'accuracy' in checkpoint_data_step1:
            print(f"  ✅ Val Accuracy: {checkpoint_data_step1['accuracy']:.3f}")
    else:
        # State dict only
        model_step1.load_state_dict(checkpoint_data_step1)
        print(f"  ✅ Loaded state_dict")
    
    # Load Step 2 checkpoint with format checking
    print(f"📥 Loading Step 2 checkpoint: {checkpoint_step2}")
    checkpoint_data_step2 = torch.load(checkpoint_step2, map_location=device)
    
    if isinstance(checkpoint_data_step2, dict) and 'model_state_dict' in checkpoint_data_step2:
        # Full checkpoint with metadata
        model_step2.load_state_dict(checkpoint_data_step2['model_state_dict'])
        print(f"  ✅ Loaded from epoch {checkpoint_data_step2.get('epoch', 'unknown')}")
        if 'accuracy' in checkpoint_data_step2:
            print(f"  ✅ Val Accuracy: {checkpoint_data_step2['accuracy']:.3f}")
    else:
        # State dict only
        model_step2.load_state_dict(checkpoint_data_step2)
        print(f"  ✅ Loaded state_dict")
    
    print("✅ Models loaded successfully!\n")
    
    # Run two-step inference
    predictions, true_labels, image_paths = two_step_inference(
        model_step1, model_step2, val_loader, device, val_dataset
    )
    
    # Calculate accuracy
    accuracy = (predictions == true_labels).mean()
    
    # Print results
    print("\n" + "="*70)
    print("📊 FINAL 3-CLASS RESULTS")
    print("="*70)
    print(f"\nOverall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)\n")
    
    print("Classification Report:")
    print(classification_report(
        true_labels, predictions,
        target_names=['Health', 'Rust', 'Other'],
        digits=3
    ))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions)
    print("           Pred: Health  Rust  Other")
    print(f"True Health:     {cm[0][0]:5d} {cm[0][1]:5d} {cm[0][2]:5d}")
    print(f"True Rust:       {cm[1][0]:5d} {cm[1][1]:5d} {cm[1][2]:5d}")
    print(f"True Other:      {cm[2][0]:5d} {cm[2][1]:5d} {cm[2][2]:5d}")
    
    print("\n" + "="*70)
    print("✅ Inference completed!")
    print("="*70 + "\n")
    
    # ==================== SAVE TO CSV ====================
    import csv
    from pathlib import Path
    
    output_csv = project_root / 'predictions.csv'
    
    label_names = {0: 'Health', 1: 'Rust', 2: 'Other'}
    
    print("\n💾 Saving predictions to CSV...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['Image_Path', 'True_Label', 'Predicted_Label', 'True_Class', 'Predicted_Class', 'Correct'])
        
        # Write predictions
        for img_path, true_label, pred_label in zip(image_paths, true_labels, predictions):
            filename = Path(img_path).name
            true_class = label_names[true_label]
            pred_class = label_names[pred_label]
            correct = 'Yes' if true_label == pred_label else 'No'
            
            writer.writerow([
                filename,
                true_label,
                pred_label,
                true_class,
                pred_class,
                correct
            ])
    
    print(f"✅ Predictions saved to: {output_csv}")
    print(f"📊 Total samples: {len(predictions)}")
    print(f"✅ Correct: {(predictions == true_labels).sum()}")
    print(f"❌ Incorrect: {(predictions != true_labels).sum()}")
    print(f"📈 Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)\n")
