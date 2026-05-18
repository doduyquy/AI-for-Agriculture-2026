import torch
import numpy as np
from sklearn.metrics import classification_report


class Evaluator:
    """
    Evaluator hỗ trợ cả 2 chế độ:
      - RGB-only   : model.forward(images)        → ResNetClassifier
      - Multimodal : model.forward(hs, ms, rgb)   → MultimodalClassifier

    Args:
        model:          Model cần đánh giá.
        val_loader:     DataLoader cho validation set.
        device:         torch.device.
        class_names:    Danh sách tên class theo thứ tự index.
        is_multimodal:  True nếu dùng MultimodalClassifier.
    """

    def __init__(self, model, val_loader, device, class_names, is_multimodal: bool = False):
        self.model         = model
        self.val_loader    = val_loader
        self.device        = device
        self.class_names   = class_names
        self.is_multimodal = is_multimodal

    def evaluate(self, model_path=None):
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded model from: {model_path}")

        self.model.eval()

        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                if self.is_multimodal:
                    hs, ms, rgb, labels = batch
                    hs  = hs.to(self.device)
                    ms  = ms.to(self.device)
                    rgb = rgb.to(self.device)
                    outputs = self.model(hs, ms, rgb)
                else:
                    if len(batch) == 4:
                        _, _, rgb, labels = batch
                    else:
                        rgb, labels = batch
                    rgb = rgb.to(self.device)
                    outputs = self.model(rgb)

                preds = outputs.argmax(dim=1).cpu().numpy()
                y_true.extend(labels.numpy())
                y_pred.extend(preds)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        report_dict = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )

        return y_true, y_pred, report_dict
