import os
import yaml
import random
import numpy as np
import torch

class ConfigDict(dict):
    """Dictionary that supports attribute-style access"""
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        self[name] = value

    @property
    def device(self):
        requested = str(self.get("DEVICE", "auto")).strip().lower()
        if requested in {"", "auto"}:
            return "cuda" if torch.cuda.is_available() else "cpu"

        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "Config đang yêu cầu DEVICE='cuda' nhưng PyTorch không thấy GPU. "
                "Trên Kaggle hãy bật Settings -> Accelerator -> GPU rồi chạy lại notebook."
            )

        if requested in {"cpu"} or requested.startswith("cuda"):
            return requested

        raise ValueError("DEVICE chỉ hỗ trợ 'auto', 'cpu', hoặc 'cuda'.")

def load_config(config_paths):
    """Load configuration from one or multiple YAML files"""
    config_dict = {}
    
    if isinstance(config_paths, str):
        config_paths = [config_paths]
        
    for yaml_path in config_paths:
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
            if cfg:
                config_dict.update(cfg)
                
    return ConfigDict(config_dict)

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def label_from_filename(fname: str) -> str:
    """Extract label from filename: 'Rust_hyper_184.png' -> 'Rust'"""
    return os.path.basename(fname).split("_")[0]

def get_filename_crossplatform(path: str) -> str:
    """Extract filename from path, works with both Windows and Linux paths"""
    return path.replace("\\\\", "/").split("/")[-1]
