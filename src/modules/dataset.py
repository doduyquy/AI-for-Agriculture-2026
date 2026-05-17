import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.modules.utils import label_from_filename

# ---------------------------------------------------------------------------
# Dataset structure (thực tế trên Kaggle):
#
#   train/RGB/
#       Health_hyper_1.png        ← label = "Health" (token đầu trước "_")
#       Rust_hyper_184.png        ← label = "Rust"
#       Other_hyper_5.png         ← label = "Other"
#       ...                       (600 files, có label trong tên)
#
#   val/RGB/
#       val_000a83c1.png          ← KHÔNG có label (competition submission set)
#       val_00a704b1.png
#       ...                       (300 files, chỉ dùng để predict & submit)
#
# Vì vậy:
#   - Validation trong quá trình train → split từ train set (20%)
#   - Submission inference           → dùng val/RGB/ với RGBTestDataset
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
    Unlabeled RGB dataset — val/RGB/ với tên file dạng 'val_<hash>.png'.
    Trả về (image, filename) để dùng cho inference và tạo submission.

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
