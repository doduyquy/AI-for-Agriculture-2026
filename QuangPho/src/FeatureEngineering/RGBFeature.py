import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
class RGBFeature:
    @staticmethod
    def glcm_features_single_channel(channel, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
        # Tính toán ma trận đồng xuất
        glcm = graycomatrix(
            channel,
            distances=distances,
            angles=angles,
            levels=levels,
            symmetric=True,
            normed=True
        )
        
        features = []
        props = ['contrast', 'homogeneity', 'correlation']

        for prop in props:
            values = graycoprops(glcm, prop)
            features.extend(values.flatten())
        return np.array(features)

    @staticmethod
    def glcm_rgb(image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None, None
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        R, G, B = cv2.split(img_rgb)

        feat_R = RGBFeature.glcm_features_single_channel(R)
        feat_G = RGBFeature.glcm_features_single_channel(G)
        feat_B = RGBFeature.glcm_features_single_channel(B)

        # Gộp thành vector đặc trưng duy nhất
        feature_vector = np.concatenate([feat_R, feat_G, feat_B])

        return feature_vector, img_rgb
    @staticmethod
    def lbp_single_channel(
        channel,
        P=8,        # số pixel lân cận
        R=1,        # bán kính
        method='uniform'
    ):
        """
        channel: ảnh grayscale (2D)
        return: histogram LBP (feature vector)
        """

        lbp = local_binary_pattern(channel, P, R, method)

        if method == 'uniform':
            n_bins = P + 2
        else:
            n_bins = 2 ** P

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=n_bins,
            range=(0, n_bins),
            density=True
        )

        return hist
    @staticmethod
    def lbp_rgb(image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        R, G, B = cv2.split(img)

        feat_R = RGBFeature.lbp_single_channel(R)
        feat_G = RGBFeature.lbp_single_channel(G)
        feat_B = RGBFeature.lbp_single_channel(B)

        feature_vector = np.concatenate([feat_R, feat_G, feat_B])
        return feature_vector
    @staticmethod
    def lbp_multiscale(channel):
        feats = []
        for R in [1, 2, 3]:
            feats.append(RGBFeature.lbp_single_channel(channel, P=8, R=R))
        return np.concatenate(feats)
    @staticmethod
    def color_moments_single_channel(channel):
        channel = channel.astype(np.float32)

        mean = np.mean(channel)
        std = np.std(channel)
        skew = np.mean((channel - mean) ** 3) / (std ** 3 + 1e-8)

        return np.array([mean, std, skew])
    @staticmethod
    def color_moments_extended(channel):
        return np.array([
            np.mean(channel),
            np.std(channel),
            np.mean((channel - np.mean(channel)) ** 3) / (np.std(channel) ** 3 + 1e-8),
            np.min(channel),
            np.max(channel)
        ])

    @staticmethod
    def color_moments_rgb(image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        features = []
        for c in cv2.split(img):
            features.extend(RGBFeature.color_moments_extended(c))

        return np.array(features)
    # color Index
    @staticmethod
    def hsv_features(image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

        features = []

        for i in range(3):  # H, S, V
            channel = img[:,:,i]
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.min(channel),
                np.max(channel)
            ])

        return np.array(features)
    @staticmethod
    def exg_index(img):
        R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
        return 2*G - R - B
    @staticmethod
    def exg_features(image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        exg = RGBFeature.exg_index(img)

        return np.array([
            np.mean(exg),
            np.std(exg),
            np.percentile(exg, 90)
        ])
    @staticmethod
    def vari_index(img):
        R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
        return (G - R) / (G + R - B + 1e-6)
    @staticmethod
    def vari_features(image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        vari = RGBFeature.vari_index(img)

        return np.array([
            np.mean(vari),
            np.std(vari),
            np.min(vari)
        ])
    @staticmethod
    def rgb_color_veg_features(image_path):
        hsv_feat = RGBFeature.hsv_features(image_path)
        exg_feat = RGBFeature.exg_features(image_path)
        vari_feat = RGBFeature.vari_features(image_path)

        return np.concatenate([hsv_feat, exg_feat, vari_feat])





# --- Chạy thử nghiệm ---
# Test GLCM RGB Feature Extraction
# --- Phần thực thi TEST ---
# path = r"D:\Kaggle\QuangPho\Data\train\RGB\Rust_hyper_1.png"

# # 1. Test GLCM
# features_glcm, img_rgb = RGBFeature.glcm_rgb(path)

# # 2. Test LBP (Lấy ví dụ trên kênh Green - kênh quan trọng của thực vật)
# if img_rgb is not None:
#     R, G, B = cv2.split(img_rgb)
#     features_lbp = RGBFeature.lbp_multiscale(G)
    
#     print(f"--- KẾT QUẢ ---")
#     print(f"Số chiều GLCM: {features_glcm.shape[0]}") # Hiện tại là 36
#     print(f"Số chiều LBP Multiscale (Kênh G): {features_lbp.shape[0]}") # 30 (10 bins * 3 scales)

#     # Hiển thị trực quan
#     plt.figure(figsize=(15, 5))
    
#     # Ảnh gốc
#     plt.subplot(1, 3, 1)
#     plt.imshow(img_rgb)
#     plt.title("Ảnh gốc (RGB)")
#     plt.axis('off')
    
#     # Biểu đồ GLCM
#     plt.subplot(1, 3, 2)
#     plt.plot(features_glcm, color='tab:blue')
#     plt.title("GLCM Vector (Texture Stats)")
#     plt.xlabel("Index")
    
#     # Biểu đồ LBP
#     plt.subplot(1, 3, 3)
#     plt.bar(range(len(features_lbp)), features_lbp, color='tab:green')
#     plt.title("LBP Multiscale (Local Patterns)")
#     plt.xlabel("Bins")
    
#     plt.tight_layout()
#     plt.show()
# else:
#     print("Lỗi: Không tìm thấy ảnh. Hãy kiểm tra lại đường dẫn!")