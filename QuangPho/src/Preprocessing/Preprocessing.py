import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import tifffile as tiff
import pandas as pd
import os
from pathlib import Path
import pickle

class Preprocessing(Dataset):
    def __init__(self, csv_file, transform=None, phase='train', pca_model_path=None):
        """
        Args:
            csv_file (str): Đường dẫn tới file CSV chứa thông tin ảnh và nhãn
            transform: Augmentation transforms (optional)
            phase (str): 'train' hoặc 'val'
            pca_model_path (str): Đường dẫn tới PCA model đã fit (optional)
        """
        self.data_info = pd.read_csv(csv_file)
        self.phase = phase
        self.transform = transform
        
        # Load PCA model nếu có
        self.pca = None
        if pca_model_path and os.path.exists(pca_model_path):
            with open(pca_model_path, 'rb') as f:
                self.pca = pickle.load(f)
    
    def __len__(self):
        return len(self.data_info)
    
    def preprocess_rgb(self, rgb_path):
        """Preprocessing cho ảnh RGB (.png)"""
        rgb_image = cv2.imread(rgb_path)
        if rgb_image is None:
            raise ValueError(f"Không thể đọc ảnh: {rgb_path}")
        
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (224, 224))
        rgb_image = rgb_image / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb_image = (rgb_image - mean) / std
        return rgb_image.transpose((2, 0, 1)).astype(np.float32)

    def preprocess_ms(self, ms_path):
        """Preprocessing cho ảnh Multispectral (.tif)"""
        img = tiff.imread(ms_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh: {ms_path}")
        
        # Đảm bảo img có shape (H, W, C)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        
        # Normalize
        img = np.clip(img, 0, 4096) / 4096.0
        
        # Tính NDVI nếu có đủ kênh (giả sử Red=2, NIR=3)
        if img.shape[2] >= 4:
            red = img[:, :, 2]
            nir = img[:, :, 3]
            ndvi = (nir - red) / (nir + red + 1e-6)
            ndvi = np.expand_dims(ndvi, axis=2)
            img = np.concatenate((img, ndvi), axis=2)
        
        # Resize
        img = cv2.resize(img, (224, 224))
        
        return img.transpose(2, 0, 1).astype(np.float32)
    
    def preprocess_hs(self, path):
        """Preprocessing cho ảnh Hyperspectral (.tif)"""
        img = tiff.imread(path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh: {path}")
        
        # Đảm bảo có 3 chiều
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        
        # Làm mượt phổ nếu có nhiều kênh
        if img.shape[2] > 11:
            try:
                img = savgol_filter(img, 11, 3, axis=2)
            except:
                pass  # Nếu lỗi thì bỏ qua bước này
        
        # PCA giảm chiều (optional)
        if self.pca is not None:
            H, W, C = img.shape
            img_flat = img.reshape(-1, C)
            img_pca = self.pca.transform(img_flat)
            img = img_pca.reshape(H, W, -1)
        
        # Resize
        img = cv2.resize(img, (112, 112))
        
        # Normalize
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
        
        return img.transpose(2, 0, 1).astype(np.float32)
    
    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        
        rgb = self.preprocess_rgb(row['rgb_path'])
        ms = self.preprocess_ms(row['ms_path'])
        hs = self.preprocess_hs(row['hs_path'])
        
        label = row['label']
        
        return {
            'rgb': torch.from_numpy(rgb),
            'ms': torch.from_numpy(ms),
            'hs': torch.from_numpy(hs),
            'label': torch.tensor(label, dtype=torch.long)
        }


def create_dataset_csv(data_root, output_csv, phase='train'):
    """
    Tạo file CSV chứa đường dẫn và nhãn cho dataset
    
    Args:
        data_root (str): Đường dẫn tới thư mục Data/
        output_csv (str): Đường dẫn output file CSV
        phase (str): 'train' hoặc 'val'
    """
    data_root = Path(data_root)
    phase_dir = data_root / phase
    
    # Mapping nhãn
    label_map = {
        'Health': 0,
        'Other': 1,
        'Rust': 2
    }
    
    data_list = []
    
    # Quét thư mục HS
    hs_dir = phase_dir / 'HS'
    for hs_file in hs_dir.glob('*.tif'):
        filename = hs_file.stem  # Tên file không có extension
        
        # Xác định nhãn từ tên file
        label_name = None
        for label in ['Health', 'Other', 'Rust']:
            if filename.startswith(label):
                label_name = label
                break
        
        if label_name is None:
            continue
        
        # Tìm file tương ứng trong MS và RGB
        ms_file = phase_dir / 'MS' / f'{filename}.tif'
        rgb_file = phase_dir / 'RGB' / f'{filename}.png'
        
        # Kiểm tra file tồn tại
        if ms_file.exists() and rgb_file.exists():
            data_list.append({
                'hs_path': str(hs_file),
                'ms_path': str(ms_file),
                'rgb_path': str(rgb_file),
                'label': label_map[label_name],
                'label_name': label_name
            })
    
    # Tạo DataFrame và lưu
    df = pd.DataFrame(data_list)
    df.to_csv(output_csv, index=False)
    
    print(f"✅ Đã tạo file {output_csv}")
    print(f"   Tổng số mẫu: {len(df)}")
    print(f"   Phân bố nhãn:")
    print(df['label_name'].value_counts())
    
    return df


def create_all_datasets(data_root='Data', output_dir='Data'):
    """
    Tạo file CSV cho cả train và validation
    
    Args:
        data_root (str): Đường dẫn tới thư mục Data/
        output_dir (str): Thư mục lưu file CSV
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🔄 Đang tạo dataset cho TRAIN...")
    train_df = create_dataset_csv(data_root, output_dir / 'train.csv', 'train')
    
    print("\n🔄 Đang tạo dataset cho VALIDATION...")
    val_df = create_dataset_csv(data_root, output_dir / 'val.csv', 'val')
    
    print("\n✅ Hoàn thành!")
    print(f"   Train: {len(train_df)} mẫu")
    print(f"   Val: {len(val_df)} mẫu")
    
    return train_df, val_df


if __name__ == '__main__':
    # Ví dụ sử dụng
    # Tạo file CSV
    script_dir = Path(__file__).parent.parent.parent  # Lên 3 cấp để về root
    data_root = script_dir / 'Data'
    
    print(f"📂 Data root: {data_root}")
    train_df, val_df = create_all_datasets(data_root, data_root)
    
    # Load dataset
    print("\n📊 Test loading dataset...")
    train_dataset = Preprocessing(
        csv_file=str(data_root / 'train.csv'),
        phase='train'
    )
    
    print(f"   Dataset size: {len(train_dataset)}")
    
    # Test lấy 1 sample
    print("\n🔍 Test lấy 1 sample...")
    sample = train_dataset[0]
    print(f"   RGB shape: {sample['rgb'].shape}")
    print(f"   MS shape: {sample['ms'].shape}")
    print(f"   HS shape: {sample['hs'].shape}")
    print(f"   Label: {sample['label']}")

