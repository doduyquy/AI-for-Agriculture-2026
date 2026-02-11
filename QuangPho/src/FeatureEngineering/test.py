import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.util import view_as_windows

def create_texture_maps(image_path, window_size=7):
    # 1. Đọc ảnh và chuyển sang mức xám (GLCM bản đồ thường dùng mức xám)
    img = cv2.imread(image_path, 0) 
    # Thu nhỏ ảnh một chút nếu ảnh quá lớn để chạy nhanh hơn
    img = cv2.resize(img, (128, 128)) 

    # 2. Tạo các cửa sổ trượt (Sliding windows)
    # window_size=7 nghĩa là tính đặc trưng cho vùng 7x7 xung quanh pixel
    pad = window_size // 2
    img_padded = np.pad(img, pad, mode='reflect')
    windows = view_as_windows(img_padded, (window_size, window_size))
    
    h, w = windows.shape[:2]
    contrast_map = np.zeros((h, w))
    homogeneity_map = np.zeros((h, w))
    energy_map = np.zeros((h, w))

    # 3. Tính GLCM cho từng cửa sổ (Quá trình này hơi tốn thời gian)
    for i in range(h):
        for j in range(w):
            window = windows[i, j]
            glcm = graycomatrix(window, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            contrast_map[i, j] = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity_map[i, j] = graycoprops(glcm, 'homogeneity')[0, 0]
            energy_map[i, j] = graycoprops(glcm, 'energy')[0, 0]

    # 4. Hiển thị kết quả giống như hình mẫu
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Gray')
    
    axes[1].imshow(contrast_map, cmap='viridis')
    axes[1].set_title('Contrast Map')
    
    axes[2].imshow(homogeneity_map, cmap='magma')
    axes[2].set_title('Homogeneity Map')
    
    axes[3].imshow(energy_map, cmap='inferno')
    axes[3].set_title('Energy Map')

    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Chạy thử
create_texture_maps(r"D:\Kaggle\QuangPho\Data\train\RGB\Rust_hyper_1.png")