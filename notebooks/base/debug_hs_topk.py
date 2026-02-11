import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tifffile as tiff

# PATHS
HS_DIR_TRAIN = r"D:\HocTap\NCKH_ThayDoNhuTai\Challenges\data\raw\Kaggle_Prepared\train\HS"
HS_DIR_TEST = r"D:\HocTap\NCKH_ThayDoNhuTai\Challenges\data\raw\Kaggle_Prepared\val\HS"
TARGET_HW = (64, 64)

# Mock Mean/Std (Approximate from notebook output for testing flow)
# In real run, we would load these, but for structure check this is fine.
# We will check if they are broadcast correctly.
MOCK_MEAN_FULL = np.ones(125, dtype=np.float64) * 300.0
MOCK_STD_FULL = np.ones(125, dtype=np.float64) * 50.0

def label_from_filename(f): return f.split('_')[0]

class HSTestDataset(Dataset):
    def __init__(self, img_dir, band_indices=None, mean=None, std=None):
        self.img_dir = img_dir
        self.files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.tif', '.tiff'))])
        self.band_indices = band_indices
        
        # Handle Mean/Std Subset (Global Stats)
        if band_indices is not None:
            self.mean = torch.tensor(mean[band_indices]).view(-1, 1, 1).float()
            self.std = torch.tensor(std[band_indices]).view(-1, 1, 1).float()
        else:
            self.mean = torch.tensor(mean).view(-1, 1, 1).float()
            self.std = torch.tensor(std).view(-1, 1, 1).float()
            
        print(f"DEBUG: Dataset initialized. Mean Shape: {self.mean.shape}, Std Shape: {self.std.shape}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        path = os.path.join(self.img_dir, fname)
        arr = tiff.imread(path).astype(np.float32)
        
        original_shape = arr.shape
        
        # Robust Dims
        if arr.ndim == 3 and arr.shape[-1] >= 120 and arr.shape[-1] <= 130:
            arr = np.transpose(arr, (2, 0, 1))
        elif arr.ndim == 2: 
            arr = arr[None, :, :]
        
        transposed_shape = arr.shape
        
        # Force 125
        c = arr.shape[0]
        if c > 125: arr = arr[:125] 
        elif c < 125:
            pad = np.zeros((125 - c, arr.shape[1], arr.shape[2]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)

        # Subset
        if self.band_indices is not None:
            arr = arr[self.band_indices]
            
        x = torch.from_numpy(arr)
        
        before_norm_mean = x.mean().item()
        
        if x.shape[1:] != TARGET_HW:
            x = F.interpolate(x.unsqueeze(0), size=TARGET_HW, mode='bilinear', align_corners=False).squeeze(0)
            
        x = (x - self.mean) / (self.std + 1e-8)
        
        return {
            'x': x,
            'fname': fname,
            'original_shape': original_shape,
            'transposed_shape': transposed_shape,
            'mean_val': x.mean().item(),
            'std_val': x.std().item(),
            'before_norm_mean': before_norm_mean
        }

def check_datasets():
    print("=== Checking Datasets ===")
    
    # create dummy top 30 indices
    top_30_indices = list(range(0, 30))
    
    print(f"\nChecking TEST Dataset with Top-30...")
    if not os.path.exists(HS_DIR_TEST):
        print(f"ERROR: Test directory not found: {HS_DIR_TEST}")
        return

    test_ds = HSTestDataset(HS_DIR_TEST, band_indices=top_30_indices, mean=MOCK_MEAN_FULL, std=MOCK_STD_FULL)
    
    if len(test_ds) == 0:
        print("ERROR: No files in test dataset.")
    else:
        item = test_ds[0]
        x = item['x']
        print(f"File: {item['fname']}")
        
        # KEY CHECKS
        is_shape_30 = (x.shape[0] == 30)
        ch0_mean = x[0].mean().item()
        ch29_mean = x[29].mean().item()
        
        print(f"RESULT: Shape_Eq_30={is_shape_30} | Ch0_Mean={ch0_mean:.4f} | Ch29_Mean={ch29_mean:.4f}")
        
        # Check if Ch0 mean is suspiciously far from 0 (normalized)
        if abs(ch0_mean) > 3.0: 
             print("ALERT: Ch0 Mean is > 3.0 stds away from training mean. Possible band mismatch!")
        
    print("\nChecking Label Mapping...")
    try:
        train_files = sorted([f for f in os.listdir(HS_DIR_TRAIN) if f.lower().endswith(('.tif', '.tiff'))])
        all_labels = sorted({label_from_filename(f) for f in train_files})
        print(f"Labels found in Train: {all_labels}")
        
        label_map = {0: 'Health', 1: 'Other', 2: 'Rust'}
        print(f"submission label_map: {label_map}")
        
        mapped_labels = [label_map[i] for i in range(len(label_map))]
        if mapped_labels == all_labels:
            print("PASSED: Label mapping matches sorted order.")
        else:
            print(f"FAILED: Label mapping mismatch! Train sorted: {all_labels}, Map: {mapped_labels}")
            
    except Exception as e:
        print(f"Error checking labels: {e}")

    print("\nChecking DataLoader Shuffle...")
    loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    print(f"DataLoader shuffle=False. First batch files:")
    for batch in loader:
        # HSTestDataset returns (x, fname) in __getitem__ normally, but I modified it for debug.
        # Let's verify what happens if we iterate.
        # Oh, the DataLoader collates the dictionary.
        print(batch['fname'])
        break

if __name__ == "__main__":
    check_datasets()
