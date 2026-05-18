import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.modules.utils import label_from_filename

# ---------------------------------------------------------------------------
# Dataset structure (thực tế trên Kaggle):
#
#   train/RGB/        train/HS/        train/MS/
#       Health_hyper_1.png / .npy / .npy  ← label = "Health"
#       Rust_hyper_184.png               ← label = "Rust"
#       ...                              (600 files, có label trong tên)
#
#   val/RGB/          val/HS/          val/MS/
#       val_000a83c1.png / .npy / .npy   ← KHÔNG có label (submission set)
#       ...                              (300 files, đủ cả 3 modality)
#
# Vì vậy:
#   - Validation trong quá trình train   → split từ train set (20%)
#   - Submission inference               → dùng val/{RGB,HS,MS}/ với MultimodalTestDataset
# ---------------------------------------------------------------------------


class RGBDataset(Dataset):
    """
    Labeled RGB dataset — train/RGB/ với tên file dạng 'ClassName_hyper_N.png'.

    Label được lấy từ token đầu tiên trước dấu '_'.
    Hỗ trợ truyền file_list để dùng subset (ví dụ train split / val split).

    Args:
        img_dir:       Thư mục chứa ảnh (train/RGB/).
        transform:     Torchvision transform.
        file_list:     Danh sách filename cụ thể (để dùng subset). Nếu None → load toàn bộ.
        class_to_idx:  Mapping label→index từ train set. Nếu None → tự xây dựng.
    """

    def __init__(self, img_dir, transform=None, file_list=None, class_to_idx=None):
        self.img_dir   = img_dir
        self.transform = transform

        # ── Collect files ────────────────────────────────────────────────────
        if file_list is not None:
            self.files = file_list
        else:
            self.files = sorted([
                f for f in os.listdir(img_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])

        # ── Build / reuse class_to_idx ───────────────────────────────────────
        raw_labels = [label_from_filename(f) for f in self.files]

        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            unique_labels = sorted(set(raw_labels))
            self.class_to_idx = {c: i for i, c in enumerate(unique_labels)}

        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        # Validate — mọi label trong split này phải nằm trong class_to_idx
        unknown = set(raw_labels) - set(self.class_to_idx.keys())
        if unknown:
            raise ValueError(
                f"[RGBDataset] Phát hiện label không hợp lệ trong '{img_dir}': {unknown}\n"
                f"Known classes: {list(self.class_to_idx.keys())}\n"
                f"Kiểm tra tên file: phải có dạng 'ClassName_<anything>.png'."
            )

        self.y = [self.class_to_idx[lbl] for lbl in raw_labels]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname  = self.files[idx]
        label  = self.y[idx]
        img    = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class RGBTestDataset(Dataset):
    """
    Unlabeled RGB-only dataset — val/RGB/ với tên file dạng 'val_<hash>.png'.
    Trả về (image, filename). Dùng khi chỉ cần infer bằng RGB branch.

    Args:
        img_dir:   Thư mục chứa ảnh (val/RGB/).
        transform: Torchvision transform.
    """

    def __init__(self, img_dir, transform=None):
        self.img_dir   = img_dir
        self.transform = transform
        self.files     = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img   = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, fname


class MultimodalTestDataset(Dataset):
    """
    Unlabeled Multimodal dataset — val/{RGB,HS,MS}/ (submission set).
    Cấu trúc giống MultimodalDataset nhưng trả về (hs, ms, rgb, filename)
    thay vì (hs, ms, rgb, label) vì tập val không có nhãn.

    Args:
        hs_dir:    Thư mục chứa dữ liệu HS (.npy).
        ms_dir:    Thư mục chứa dữ liệu MS (.npy hoặc .tif).
        rgb_dir:   Thư mục chứa ảnh RGB (.png/.jpg).
        transform: Torchvision transform (áp dụng cho RGB).
    """

    def __init__(self, hs_dir, ms_dir, rgb_dir, transform=None):
        self.hs_dir    = hs_dir
        self.ms_dir    = ms_dir
        self.rgb_dir   = rgb_dir
        self.transform = transform

        # Dùng rgb_dir làm nguồn chính để lấy danh sách file
        self.files = sorted([
            f for f in os.listdir(rgb_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname     = self.files[idx]
        base_name = os.path.splitext(fname)[0]

        # 1. Load RGB (3 channels)
        rgb_path = os.path.join(self.rgb_dir, fname)
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(self.rgb_dir, base_name + '.png')
        try:
            rgb_img = Image.open(rgb_path).convert("RGB")
            if self.transform:
                rgb_img = self.transform(rgb_img)
            else:
                rgb_img = transforms.ToTensor()(rgb_img)
        except Exception:
            rgb_img = torch.zeros((3, 64, 64), dtype=torch.float32)

        # 2. Load HS (125 channels, 32x32)
        hs_path = os.path.join(self.hs_dir, base_name + '.npy')
        if os.path.exists(hs_path):
            hs_data = np.load(hs_path)
            if len(hs_data.shape) == 3 and hs_data.shape[-1] == 125:
                hs_data = hs_data.transpose(2, 0, 1)
            hs_tensor = torch.tensor(hs_data, dtype=torch.float32)
        else:
            hs_tensor = torch.zeros((125, 32, 32), dtype=torch.float32)

        # 3. Load MS (5 channels, 64x64)
        ms_path_npy = os.path.join(self.ms_dir, base_name + '.npy')
        ms_path_tif = os.path.join(self.ms_dir, base_name + '.tif')
        if os.path.exists(ms_path_npy):
            ms_data = np.load(ms_path_npy)
            if len(ms_data.shape) == 3 and ms_data.shape[-1] == 5:
                ms_data = ms_data.transpose(2, 0, 1)
            ms_tensor = torch.tensor(ms_data, dtype=torch.float32)
        elif os.path.exists(ms_path_tif):
            try:
                ms_img = Image.open(ms_path_tif)
                ms_tensor = transforms.ToTensor()(ms_img)
            except Exception:
                ms_tensor = torch.zeros((5, 64, 64), dtype=torch.float32)
        else:
            ms_tensor = torch.zeros((5, 64, 64), dtype=torch.float32)

        return hs_tensor, ms_tensor, rgb_img, fname


def split_dataset(img_dir, val_split=0.2, seed=42):
    """
    Chia file list trong img_dir thành (train_files, val_files) theo tỉ lệ.

    Đảm bảo stratified: mỗi class được chia theo tỉ lệ val_split.
    Trả về 2 list filename (chưa load ảnh) và class_to_idx.

    Args:
        img_dir:   Thư mục chứa ảnh dạng 'ClassName_xxx.png'.
        val_split: Tỉ lệ validation (mặc định 0.2 = 20%).
        seed:      Random seed.

    Returns:
        train_files, val_files, class_to_idx
    """
    rng = random.Random(seed)

    all_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    # Group by class
    class_files: dict[str, list] = {}
    for f in all_files:
        cls = label_from_filename(f)
        class_files.setdefault(cls, []).append(f)

    train_files, val_files = [], []
    for cls, files in sorted(class_files.items()):
        shuffled = files[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_split))
        val_files.extend(shuffled[:n_val])
        train_files.extend(shuffled[n_val:])

    # Build class_to_idx từ toàn bộ tập (không chỉ train)
    all_labels = sorted(class_files.keys())
    class_to_idx = {c: i for i, c in enumerate(all_labels)}

    return train_files, val_files, class_to_idx


def get_transforms(cfg):
    """Returns training and validation transforms."""
    tfm_train = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
    ])

    tfm_val = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
    ])

    return tfm_train, tfm_val


class MultimodalDataset(Dataset):
    """
    Multimodal Dataset cho ảnh Hyperspectral (HS), Multispectral (MS) và RGB.
    
    Theo cấu trúc dự kiến:
    - HS:  125 kênh, size 32x32
    - MS:  5 kênh, size 64x64
    - RGB: 3 kênh, size 64x64
    
    Args:
        hs_dir:        Thư mục chứa ảnh HS.
        ms_dir:        Thư mục chứa ảnh MS.
        rgb_dir:       Thư mục chứa ảnh RGB.
        transform:     Torchvision transform (áp dụng cho RGB).
        file_list:     Danh sách filename dùng làm cơ sở (base name). Nếu None → duyệt rgb_dir.
        class_to_idx:  Mapping label→index.
    """

    def __init__(self, hs_dir, ms_dir, rgb_dir, transform=None, file_list=None, class_to_idx=None):
        self.hs_dir = hs_dir
        self.ms_dir = ms_dir
        self.rgb_dir = rgb_dir
        self.transform = transform

        # ── Collect files ────────────────────────────────────────────────────
        if file_list is not None:
            self.files = file_list
        else:
            # Lấy rgb_dir làm gốc chuẩn cho danh sách file
            self.files = sorted([
                f for f in os.listdir(rgb_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".npy"))
            ])

        # ── Build / reuse class_to_idx ───────────────────────────────────────
        raw_labels = [label_from_filename(f) for f in self.files]

        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            unique_labels = sorted(set(raw_labels))
            self.class_to_idx = {c: i for i, c in enumerate(unique_labels)}

        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        self.y = [self.class_to_idx[lbl] for lbl in raw_labels]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        label = self.y[idx]
        base_name = os.path.splitext(fname)[0]

        # 1. Load RGB (3 channels, 64x64)
        rgb_path = os.path.join(self.rgb_dir, fname)
        # Dự phòng trường hợp thư mục có file tên khác extension
        if not os.path.exists(rgb_path):
            rgb_path = os.path.join(self.rgb_dir, base_name + '.png')
            
        try:
            rgb_img = Image.open(rgb_path).convert("RGB")
            if self.transform:
                rgb_img = self.transform(rgb_img)
            else:
                rgb_img = transforms.ToTensor()(rgb_img)
        except Exception:
            # Dummy tensor cho RGB nếu không đọc được
            rgb_img = torch.zeros((3, 64, 64), dtype=torch.float32)

        # 2. Load HS (125 channels, 32x32)
        # Giả định dữ liệu HS được lưu dưới dạng .npy
        hs_path = os.path.join(self.hs_dir, base_name + '.npy')
        if os.path.exists(hs_path):
            hs_data = np.load(hs_path)
            # Chuyển (H, W, C) sang (C, H, W) nếu kênh nằm cuối
            if len(hs_data.shape) == 3 and hs_data.shape[-1] == 125:
                hs_data = hs_data.transpose(2, 0, 1)
            hs_tensor = torch.tensor(hs_data, dtype=torch.float32)
        else:
            # Dummy tensor cho HS
            hs_tensor = torch.zeros((125, 32, 32), dtype=torch.float32)

        # 3. Load MS (5 channels, 64x64)
        # Giả định MS được lưu dưới dạng .npy hoặc .tif
        ms_path_npy = os.path.join(self.ms_dir, base_name + '.npy')
        ms_path_tif = os.path.join(self.ms_dir, base_name + '.tif')
        
        if os.path.exists(ms_path_npy):
            ms_data = np.load(ms_path_npy)
            if len(ms_data.shape) == 3 and ms_data.shape[-1] == 5:
                ms_data = ms_data.transpose(2, 0, 1)
            ms_tensor = torch.tensor(ms_data, dtype=torch.float32)
        elif os.path.exists(ms_path_tif):
            try:
                ms_img = Image.open(ms_path_tif)
                ms_tensor = transforms.ToTensor()(ms_img)
            except Exception:
                ms_tensor = torch.zeros((5, 64, 64), dtype=torch.float32)
        else:
            # Dummy tensor cho MS
            ms_tensor = torch.zeros((5, 64, 64), dtype=torch.float32)

        return hs_tensor, ms_tensor, rgb_img, label
