"""
PipelineV1_Ensemble.py - ENSEMBLE MULTIPLE MODELS FOR STEP 1

🎯 Strategy:
  - Train 3 different models cho Step 1:
    1. RGB-Dominant Model (85% RGB) - Tốt cho cây khỏe
    2. Spectral-Dominant Model (70% Spectral) - Tốt cho diseased
    3. Balanced Model (50% RGB, 50% Spectral) - Cân bằng
  
  - Ensemble Voting:
    * CONSERVATIVE: Nếu ≥2 models predict Diseased → Final = Diseased
    * Target: Minimize Diseased→Health (FN) errors!

✅ Kỳ vọng:
  - Diseased→Health: 22.5% → <10%
  - Overall Accuracy: 67.5% → 75%+
  - Kaggle Score: 0.56 → 0.70-0.75
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# Import from existing pipeline
import importlib.util
spec = importlib.util.spec_from_file_location(
    "pipeline",
    str(project_root / "src" / "model" / "Pipeline" / "PipelineV1_Stage1_TwoStep.py")
)
pipeline_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_module)

# Use existing classes
TwoStepWheatNet = pipeline_module.TwoStepWheatNet
Stage1Dataset = pipeline_module.Stage1Dataset
load_data_from_folder = pipeline_module.load_data_from_folder
FocalLoss = pipeline_module.FocalLoss


# ==================== ENSEMBLE VOTER ====================
class Step1Ensemble:
    """
    Ensemble 3 models với voting strategy khác nhau
    
    Models:
      1. RGB-Dominant (85% RGB) - Giỏi nhận diện cây khỏe
      2. Spectral-Dominant (70% Spectral) - Giỏi nhận diện bệnh
      3. Balanced (50-50) - Cân bằng
    
    Voting:
      - CONSERVATIVE: ≥2 models predict Diseased → Diseased
      - AGGRESSIVE: Chỉ cần 1 model predict Diseased → Diseased
      - SOFT: Average probabilities
    """
    def __init__(self, model_rgb, model_spectral, model_balanced, mode='conservative'):
        self.model_rgb = model_rgb
        self.model_spectral = model_spectral
        self.model_balanced = model_balanced
        self.mode = mode
        
        print("\n" + "="*70)
        print("🔀 STEP 1 ENSEMBLE INITIALIZED")
        print("="*70)
        print(f"  Model 1: RGB-Dominant (85% RGB)")
        print(f"  Model 2: Spectral-Dominant (70% Spectral)")
        print(f"  Model 3: Balanced (50% RGB, 50% Spectral)")
        print(f"  Voting Mode: {mode.upper()}")
        if mode == 'conservative':
            print(f"    → ≥2 models predict Diseased → Final = Diseased")
            print(f"    → Target: Minimize Diseased→Health errors!")
        elif mode == 'aggressive':
            print(f"    → ≥1 model predicts Diseased → Final = Diseased")
            print(f"    → Ultra-conservative, catch ALL diseased!")
        else:
            print(f"    → Soft voting: Average probabilities")
        print("="*70 + "\n")
    
    @torch.no_grad()
    def predict(self, rgb, ms, hs, handcrafted, device):
        """
        Ensemble prediction
        
        Returns:
            predictions: (B,) binary labels [0=Healthy, 1=Diseased]
            confidences: (B,) confidence scores [0-1]
        """
        self.model_rgb.eval()
        self.model_spectral.eval()
        self.model_balanced.eval()
        
        # Get predictions from all 3 models
        out_rgb = self.model_rgb(rgb, ms, hs, rgb_handcrafted=handcrafted, step='step1')
        out_spec = self.model_spectral(rgb, ms, hs, rgb_handcrafted=handcrafted, step='step1')
        out_bal = self.model_balanced(rgb, ms, hs, rgb_handcrafted=handcrafted, step='step1')
        
        # Convert to probabilities
        prob_rgb = F.softmax(out_rgb, dim=1)
        prob_spec = F.softmax(out_spec, dim=1)
        prob_bal = F.softmax(out_bal, dim=1)
        
        if self.mode == 'conservative':
            # Conservative: ≥2 models predict Diseased
            pred_rgb = (prob_rgb[:, 1] > 0.5).long()
            pred_spec = (prob_spec[:, 1] > 0.5).long()
            pred_bal = (prob_bal[:, 1] > 0.5).long()
            
            # Count votes for Diseased
            votes = pred_rgb + pred_spec + pred_bal
            predictions = (votes >= 2).long()  # ≥2 votes → Diseased
            
            # Confidence = average of P(Diseased) from voting models
            confidences = (prob_rgb[:, 1] + prob_spec[:, 1] + prob_bal[:, 1]) / 3
            
        elif self.mode == 'aggressive':
            # Aggressive: ≥1 model predicts Diseased
            pred_rgb = (prob_rgb[:, 1] > 0.5).long()
            pred_spec = (prob_spec[:, 1] > 0.5).long()
            pred_bal = (prob_bal[:, 1] > 0.5).long()
            
            votes = pred_rgb + pred_spec + pred_bal
            predictions = (votes >= 1).long()  # ≥1 vote → Diseased
            
            confidences = (prob_rgb[:, 1] + prob_spec[:, 1] + prob_bal[:, 1]) / 3
            
        else:  # soft voting
            # Average probabilities
            avg_prob = (prob_rgb + prob_spec + prob_bal) / 3
            predictions = avg_prob.argmax(1)
            confidences = avg_prob[:, predictions]
        
        return predictions, confidences


# ==================== TRAINING FUNCTION ====================
def train_ensemble_models(train_loader, val_loader, device):
    """
    Train 3 models với RGB weights khác nhau
    """
    print("\n" + "="*70)
    print("🎯 TRAINING ENSEMBLE MODELS")
    print("="*70 + "\n")
    
    # ===== MODEL 1: RGB-DOMINANT (85% RGB) =====
    print("📊 Model 1: RGB-Dominant (85% RGB, 15% Spectral)")
    print("  - Giỏi nhận diện cây khỏe (visual appearance)")
    print("  - Focus vào chlorophyll, green color\n")
    
    model_rgb = TwoStepWheatNet(dropout_rate=0.4).to(device)
    # Modify Step 1 gate weight
    model_rgb.step1_gate.rgb_weight = 0.85
    model_rgb.step1_gate.spec_weight = 0.15
    
    optimizer_rgb = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_rgb.parameters()),
        lr=1e-4, weight_decay=1e-3
    )
    criterion_rgb = FocalLoss(alpha=3.0, gamma=2.0)
    
    best_acc_rgb = 0.0
    for epoch in range(1, 21):
        # Training
        model_rgb.train()
        for rgb, ms, hs, handcrafted, labels in train_loader:
            rgb, ms, hs, handcrafted, labels = rgb.to(device), ms.to(device), hs.to(device), handcrafted.to(device), labels.to(device)
            labels_binary = (labels > 0).long()
            
            optimizer_rgb.zero_grad()
            outputs = model_rgb(rgb, ms, hs, rgb_handcrafted=handcrafted, step='step1')
            loss = criterion_rgb(outputs, labels_binary)
            loss.backward()
            optimizer_rgb.step()
        
        # Validation
        model_rgb.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for rgb, ms, hs, handcrafted, labels in val_loader:
                rgb, ms, hs, handcrafted = rgb.to(device), ms.to(device), hs.to(device), handcrafted.to(device)
                labels_binary = (labels > 0).long()
                
                outputs = model_rgb(rgb, ms, hs, rgb_handcrafted=handcrafted, step='step1')
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.cpu().eq(labels_binary).sum().item()
        
        val_acc = correct / total
        if val_acc > best_acc_rgb:
            best_acc_rgb = val_acc
            torch.save(model_rgb.state_dict(), 'checkpoints/ensemble_rgb_model.pth')
        
        if epoch % 5 == 0:
            print(f"  [Epoch {epoch:2d}] Val Acc: {val_acc:.3f} (Best: {best_acc_rgb:.3f})")
    
    print(f"✅ Model 1 trained! Best Acc: {best_acc_rgb:.3f}\n")
    
    # ===== MODEL 2: SPECTRAL-DOMINANT (30% RGB, 70% Spectral) =====
    print("📊 Model 2: Spectral-Dominant (30% RGB, 70% Spectral)")
    print("  - Giỏi nhận diện bệnh (spectral signatures)")
    print("  - Focus vào MS/HS bands\n")
    
    model_spectral = TwoStepWheatNet(dropout_rate=0.4).to(device)
    model_spectral.step1_gate.rgb_weight = 0.30
    model_spectral.step1_gate.spec_weight = 0.70
    
    optimizer_spec = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_spectral.parameters()),
        lr=1e-4, weight_decay=1e-3
    )
    criterion_spec = FocalLoss(alpha=3.0, gamma=2.0)
    
    best_acc_spec = 0.0
    for epoch in range(1, 21):
        # Training
        model_spectral.train()
        for rgb, ms, hs, handcrafted, labels in train_loader:
            rgb, ms, hs, handcrafted, labels = rgb.to(device), ms.to(device), hs.to(device), handcrafted.to(device), labels.to(device)
            labels_binary = (labels > 0).long()
            
            optimizer_spec.zero_grad()
            outputs = model_spectral(rgb, ms, hs, rgb_handcrafted=handcrafted, step='step1')
            loss = criterion_spec(outputs, labels_binary)
            loss.backward()
            optimizer_spec.step()
        
        # Validation
        model_spectral.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for rgb, ms, hs, handcrafted, labels in val_loader:
                rgb, ms, hs, handcrafted = rgb.to(device), ms.to(device), hs.to(device), handcrafted.to(device)
                labels_binary = (labels > 0).long()
                
                outputs = model_spectral(rgb, ms, hs, rgb_handcrafted=handcrafted, step='step1')
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.cpu().eq(labels_binary).sum().item()
        
        val_acc = correct / total
        if val_acc > best_acc_spec:
            best_acc_spec = val_acc
            torch.save(model_spectral.state_dict(), 'checkpoints/ensemble_spectral_model.pth')
        
        if epoch % 5 == 0:
            print(f"  [Epoch {epoch:2d}] Val Acc: {val_acc:.3f} (Best: {best_acc_spec:.3f})")
    
    print(f"✅ Model 2 trained! Best Acc: {best_acc_spec:.3f}\n")
    
    # ===== MODEL 3: BALANCED (50% RGB, 50% Spectral) =====
    print("📊 Model 3: Balanced (50% RGB, 50% Spectral)")
    print("  - Cân bằng cả 2 modalities")
    print("  - Không thiên vị\n")
    
    model_balanced = TwoStepWheatNet(dropout_rate=0.4).to(device)
    model_balanced.step1_gate.rgb_weight = 0.50
    model_balanced.step1_gate.spec_weight = 0.50
    
    optimizer_bal = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_balanced.parameters()),
        lr=1e-4, weight_decay=1e-3
    )
    criterion_bal = FocalLoss(alpha=3.0, gamma=2.0)
    
    best_acc_bal = 0.0
    for epoch in range(1, 21):
        # Training
        model_balanced.train()
        for rgb, ms, hs, handcrafted, labels in train_loader:
            rgb, ms, hs, handcrafted, labels = rgb.to(device), ms.to(device), hs.to(device), handcrafted.to(device), labels.to(device)
            labels_binary = (labels > 0).long()
            
            optimizer_bal.zero_grad()
            outputs = model_balanced(rgb, ms, hs, rgb_handcrafted=handcrafted, step='step1')
            loss = criterion_bal(outputs, labels_binary)
            loss.backward()
            optimizer_bal.step()
        
        # Validation
        model_balanced.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for rgb, ms, hs, handcrafted, labels in val_loader:
                rgb, ms, hs, handcrafted = rgb.to(device), ms.to(device), hs.to(device), handcrafted.to(device)
                labels_binary = (labels > 0).long()
                
                outputs = model_balanced(rgb, ms, hs, rgb_handcrafted=handcrafted, step='step1')
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.cpu().eq(labels_binary).sum().item()
        
        val_acc = correct / total
        if val_acc > best_acc_bal:
            best_acc_bal = val_acc
            torch.save(model_balanced.state_dict(), 'checkpoints/ensemble_balanced_model.pth')
        
        if epoch % 5 == 0:
            print(f"  [Epoch {epoch:2d}] Val Acc: {val_acc:.3f} (Best: {best_acc_bal:.3f})")
    
    print(f"✅ Model 3 trained! Best Acc: {best_acc_bal:.3f}\n")
    
    # Load best models
    model_rgb.load_state_dict(torch.load('checkpoints/ensemble_rgb_model.pth'))
    model_spectral.load_state_dict(torch.load('checkpoints/ensemble_spectral_model.pth'))
    model_balanced.load_state_dict(torch.load('checkpoints/ensemble_balanced_model.pth'))
    
    return model_rgb, model_spectral, model_balanced


# ==================== EVALUATE ENSEMBLE ====================
@torch.no_grad()
def evaluate_ensemble(ensemble, val_loader, device):
    """Evaluate ensemble performance"""
    all_preds = []
    all_labels = []
    all_confidences = []
    
    for rgb, ms, hs, handcrafted, labels in val_loader:
        rgb = rgb.to(device)
        ms = ms.to(device)
        hs = hs.to(device)
        handcrafted = handcrafted.to(device)
        labels_binary = (labels > 0).long()
        
        predictions, confidences = ensemble.predict(rgb, ms, hs, handcrafted, device)
        
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels_binary.numpy())
        all_confidences.extend(confidences.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    acc = (all_preds == all_labels).mean()
    
    # Step 1 specific metrics
    diseased_mask = (all_labels == 1)
    healthy_mask = (all_labels == 0)
    
    # False Negative (Diseased→Health)
    fn_mask = (all_labels == 1) & (all_preds == 0)
    fn_rate = fn_mask.sum() / diseased_mask.sum() if diseased_mask.sum() > 0 else 0
    
    # False Positive (Health→Diseased)
    fp_mask = (all_labels == 0) & (all_preds == 1)
    fp_rate = fp_mask.sum() / healthy_mask.sum() if healthy_mask.sum() > 0 else 0
    
    # Recalls
    diseased_recall = ((all_labels == 1) & (all_preds == 1)).sum() / diseased_mask.sum() if diseased_mask.sum() > 0 else 0
    healthy_recall = ((all_labels == 0) & (all_preds == 0)).sum() / healthy_mask.sum() if healthy_mask.sum() > 0 else 0
    
    return {
        'accuracy': acc,
        'fn_rate': fn_rate,
        'fp_rate': fp_rate,
        'diseased_recall': diseased_recall,
        'healthy_recall': healthy_recall,
        'predictions': all_preds,
        'labels': all_labels
    }


# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 ENSEMBLE TRAINING FOR STEP 1")
    print("="*70)
    print("Strategy: Train 3 models + Conservative Voting")
    print("  ✅ Model 1: RGB-Dominant (85% RGB)")
    print("  ✅ Model 2: Spectral-Dominant (70% Spectral)")
    print("  ✅ Model 3: Balanced (50-50)")
    print("  ✅ Voting: ≥2 models → Diseased")
    print("="*70 + "\n")
    
    # Load data
    data_root = Path(project_root) / 'Data'
    
    print("Loading data...")
    train_rgb_paths, train_ms_paths, train_hs_paths, train_labels = load_data_from_folder(data_root, 'train')
    val_rgb_paths, val_ms_paths, val_hs_paths, val_labels = load_data_from_folder(data_root, 'val')
    
    if len(val_labels) < 50:
        print("Splitting train data...")
        from sklearn.model_selection import train_test_split
        train_rgb_paths, val_rgb_paths, train_ms_paths, val_ms_paths, \
        train_hs_paths, val_hs_paths, train_labels, val_labels = train_test_split(
            train_rgb_paths, train_ms_paths, train_hs_paths, train_labels,
            test_size=0.2, random_state=42, stratify=train_labels
        )
    
    print(f"Train: {len(train_labels)} samples")
    print(f"Val: {len(val_labels)} samples\n")
    
    # Create datasets
    from torch.utils.data import WeightedRandomSampler
    
    train_dataset = Stage1Dataset(train_rgb_paths, train_ms_paths, train_hs_paths, train_labels, augment=True)
    val_dataset = Stage1Dataset(val_rgb_paths, val_ms_paths, val_hs_paths, val_labels, augment=False)
    
    # Weighted sampler (oversample Healthy 3x)
    sample_weights = [3.0 if label == 0 else 1.0 for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Train 3 models
    model_rgb, model_spectral, model_balanced = train_ensemble_models(train_loader, val_loader, device)
    
    # Evaluate individual models
    print("\n" + "="*70)
    print("📊 INDIVIDUAL MODEL PERFORMANCE")
    print("="*70 + "\n")
    
    for name, model in [("RGB-Dominant", model_rgb), ("Spectral-Dominant", model_spectral), ("Balanced", model_balanced)]:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for rgb, ms, hs, handcrafted, labels in val_loader:
                rgb, ms, hs, handcrafted = rgb.to(device), ms.to(device), hs.to(device), handcrafted.to(device)
                labels_binary = (labels > 0).long()
                
                outputs = model(rgb, ms, hs, rgb_handcrafted=handcrafted, step='step1')
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.cpu().eq(labels_binary).sum().item()
        
        print(f"{name:20s}: {correct/total:.3f}")
    
    # Evaluate ensembles
    print("\n" + "="*70)
    print("🔀 ENSEMBLE PERFORMANCE")
    print("="*70 + "\n")
    
    for mode in ['conservative', 'aggressive', 'soft']:
        ensemble = Step1Ensemble(model_rgb, model_spectral, model_balanced, mode=mode)
        results = evaluate_ensemble(ensemble, val_loader, device)
        
        print(f"\n{mode.upper()} Voting:")
        print(f"  Overall Accuracy: {results['accuracy']:.3f}")
        print(f"  Diseased→Health (FN): {results['fn_rate']:.3f} ← MINIMIZE!")
        print(f"  Health→Diseased (FP): {results['fp_rate']:.3f}")
        print(f"  Diseased Recall: {results['diseased_recall']:.3f}")
        print(f"  Healthy Recall: {results['healthy_recall']:.3f}")
        
        # Classification report
        print(f"\n  Classification Report:")
        print(classification_report(
            results['labels'], results['predictions'],
            target_names=['Healthy', 'Diseased'],
            digits=3
        ))
        
        cm = confusion_matrix(results['labels'], results['predictions'])
        print(f"  Confusion Matrix:")
        print(f"          Pred: Healthy  Diseased")
        print(f"  True Healthy : {cm[0][0]:5d} {cm[0][1]:5d}")
        print(f"  True Diseased: {cm[1][0]:5d} {cm[1][1]:5d}")
    
    print("\n" + "="*70)
    print("✅ ENSEMBLE TRAINING COMPLETE!")
    print("="*70)
    print("\nBest approach: CONSERVATIVE voting")
    print("  → ≥2 models predict Diseased → Final = Diseased")
    print("  → Expected: FN <10%, Overall Acc >75%")
    print("  → Kaggle Score: 0.70-0.75")
    print("\nNext: Update Predict_Unlabeled.py to use ensemble!")
    print("="*70 + "\n")
