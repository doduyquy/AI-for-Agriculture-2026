import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.modules.utils import label_from_filename


def _is_imagefolder_structure(img_dir: str) -> bool:
    """Check if img_dir uses ImageFolder structure (class subfolders contain images)."""
    entries = os.listdir(img_dir)
    for entry in entries:
        entry_path = os.path.join(img_dir, entry)
        if os.path.isdir(entry_path):
            # If there's at least one subfolder, treat as ImageFolder
            return True
    return False


def _load_imagefolder(img_dir: str):
    """
    Load (relative_path, class_name) pairs from an ImageFolder-style directory:
    img_dir/
        ClassName1/
            img1.png
        ClassName2/
            img2.png
    Returns:
        files: list of relative paths (e.g. 'ClassName1/img1.png')
        labels: list of class name strings
    """
    files, labels = [], []
    class_names = sorted([
        d for d in os.listdir(img_dir)
        if os.path.isdir(os.path.join(img_dir, d))
    ])
    for cls in class_names:
        cls_dir = os.path.join(img_dir, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                files.append(os.path.join(cls, fname))  # relative to img_dir
                labels.append(cls)
    return files, labels


class RGBDataset(Dataset):
    """
    RGB Image Dataset.

    Supports two directory structures automatically:
    1. Flat files with label-encoded filenames: 'ClassName_xxx.png'
       Example: Rust_hyper_184.png  →  label = 'Rust'
    2. ImageFolder structure (class subfolders):
       img_dir/ClassName1/img1.png, img_dir/ClassName2/img2.png

    Args:
        img_dir: Root image directory.
        transform: Optional torchvision transform.
        file_list: Optional explicit list of filenames (flat mode only).
        class_to_idx: Optional pre-built label→index mapping (from train set).
    """

    def __init__(self, img_dir, transform=None, file_list=None, class_to_idx=None):
        self.img_dir = img_dir
        self.transform = transform

        # ── Auto-detect structure ──────────────────────────────────────────
        if file_list is not None:
            # Explicit file list → always flat mode
            self.files = file_list
            raw_labels = [label_from_filename(f) for f in self.files]
        elif _is_imagefolder_structure(img_dir):
            # ImageFolder structure
            self.files, raw_labels = _load_imagefolder(img_dir)
        else:
            # Flat files, label from filename convention
            self.files = sorted([
                f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
            raw_labels = [label_from_filename(f) for f in self.files]

        # ── Build / reuse class_to_idx ─────────────────────────────────────
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
            # Validate that all labels in this split exist in the provided mapping
            unknown = set(raw_labels) - set(self.class_to_idx.keys())
            if unknown:
                raise ValueError(
                    f"[RGBDataset] Found labels in '{img_dir}' that are NOT in "
                    f"class_to_idx: {unknown}.\n"
                    f"Known classes: {list(self.class_to_idx.keys())}\n"
                    f"Hint: Check that the val/test directory follows the same "
                    f"naming convention as train."
                )
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

        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


class RGBTestDataset(Dataset):
    """
    RGB Test Dataset — returns (image, filename) without label.
    Supports both flat-file and ImageFolder directory structures.
    """

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        if _is_imagefolder_structure(img_dir):
            # Gather all images across subfolders; ignore subfolder name (no label needed)
            self.files = []
            for subdir in sorted(os.listdir(img_dir)):
                subdir_path = os.path.join(img_dir, subdir)
                if os.path.isdir(subdir_path):
                    for fname in sorted(os.listdir(subdir_path)):
                        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                            self.files.append(os.path.join(subdir, fname))
        else:
            self.files = sorted([
                f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, fname


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
