"""
check_checkpoints.py - Kiểm tra metadata của checkpoints

Mục đích: Xem checkpoint lưu gì, epoch nào, accuracy bao nhiêu
"""

import torch
from pathlib import Path

print("\n" + "="*70)
print("🔍 KIỂM TRA CHECKPOINTS")
print("="*70)

# Check Step 1
checkpoint_path_step1 = Path('checkpoints/best_model_step1.pth')
if not checkpoint_path_step1.exists():
    print(f"\n❌ {checkpoint_path_step1} không tồn tại!")
else:
    print(f"\n📦 Step 1 Checkpoint: {checkpoint_path_step1}")
    checkpoint_step1 = torch.load(checkpoint_path_step1, map_location='cpu')
    
    if isinstance(checkpoint_step1, dict):
        # Check for metadata
        if 'epoch' in checkpoint_step1:
            print(f"  ✅ Epoch: {checkpoint_step1['epoch']}")
        else:
            print(f"  ⚠️ No epoch metadata")
            
        if 'accuracy' in checkpoint_step1:
            print(f"  ✅ Accuracy: {checkpoint_step1['accuracy']:.3f}")
        else:
            print(f"  ⚠️ No accuracy metadata")
            
        if 'loss' in checkpoint_step1:
            print(f"  ✅ Loss: {checkpoint_step1['loss']:.3f}")
        else:
            print(f"  ⚠️ No loss metadata")
        
        # Check state_dict
        if 'model_state_dict' in checkpoint_step1:
            print(f"  ✅ Format: Full checkpoint (with model_state_dict key)")
            state_dict = checkpoint_step1['model_state_dict']
        else:
            print(f"  ⚠️ Format: state_dict only (no metadata)")
            state_dict = checkpoint_step1
        
        print(f"  📊 Total parameter keys: {len(state_dict)}")
        print(f"  📝 First 5 parameter keys:")
        for i, key in enumerate(list(state_dict.keys())[:5]):
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
            print(f"    {i+1}. {key}: {shape}")
    else:
        print("  ❌ Not a dict! Checkpoint format không đúng!")

# Check Step 2
print("\n" + "-"*70)
checkpoint_path_step2 = Path('checkpoints/best_model_step2.pth')
if not checkpoint_path_step2.exists():
    print(f"\n❌ {checkpoint_path_step2} không tồn tại!")
else:
    print(f"\n📦 Step 2 Checkpoint: {checkpoint_path_step2}")
    checkpoint_step2 = torch.load(checkpoint_path_step2, map_location='cpu')
    
    if isinstance(checkpoint_step2, dict):
        # Check for metadata
        if 'epoch' in checkpoint_step2:
            print(f"  ✅ Epoch: {checkpoint_step2['epoch']}")
        else:
            print(f"  ⚠️ No epoch metadata")
            
        if 'accuracy' in checkpoint_step2:
            print(f"  ✅ Accuracy: {checkpoint_step2['accuracy']:.3f}")
        else:
            print(f"  ⚠️ No accuracy metadata")
            
        if 'loss' in checkpoint_step2:
            print(f"  ✅ Loss: {checkpoint_step2['loss']:.3f}")
        else:
            print(f"  ⚠️ No loss metadata")
        
        # Check state_dict
        if 'model_state_dict' in checkpoint_step2:
            print(f"  ✅ Format: Full checkpoint (with model_state_dict key)")
            state_dict = checkpoint_step2['model_state_dict']
        else:
            print(f"  ⚠️ Format: state_dict only (no metadata)")
            state_dict = checkpoint_step2
        
        print(f"  📊 Total parameter keys: {len(state_dict)}")
        print(f"  📝 First 5 parameter keys:")
        for i, key in enumerate(list(state_dict.keys())[:5]):
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
            print(f"    {i+1}. {key}: {shape}")
    else:
        print("  ❌ Not a dict! Checkpoint format không đúng!")

print("\n" + "="*70)
print("✅ Check completed!")
print("="*70 + "\n")
