import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import cv2
import numpy as np
import sys
import os

# Add project root to path for imports
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.FeatureEngineering.RGBFeature import RGBFeature
from src.Preprocessing.PreRGB import cnn_transform

class WheatRGBDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        # --- RGB image ---
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_tensor = cnn_transform(img)

        # --- Handcrafted RGB features ---
        glcm_feat, _ = RGBFeature.glcm_rgb(path)      # texture thống kê
        lbp_feat = RGBFeature.lbp_multiscale(img[:,:,1])  # texture cục bộ (G)
        color_feat = RGBFeature.rgb_color_veg_features(path)  # HSV + ExG + VARI

        F_tex_spatial = np.concatenate([glcm_feat, lbp_feat])
        F_color = color_feat

        F_handcrafted = np.concatenate([F_tex_spatial, F_color])
        F_handcrafted = torch.tensor(F_handcrafted, dtype=torch.float32)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img_tensor, F_handcrafted, label
