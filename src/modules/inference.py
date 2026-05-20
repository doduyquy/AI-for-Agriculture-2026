# pyrefly: ignore [missing-import]
import torch
import pandas as pd
import os


class Inferencer:
    """
    Inferencer hỗ trợ cả 2 chế độ:
      - RGB-only   : model.forward(images)        → ResNetClassifier
                     test_loader từ RGBTestDataset → batch: (rgb, fname)
      - Multimodal : model.forward(hs, ms, rgb)   → MultimodalClassifier
                     test_loader từ MultimodalTestDataset → batch: (hs, ms, rgb, fname)

    Args:
        model:          Model đã train.
        test_loader:    DataLoader từ RGBTestDataset hoặc MultimodalTestDataset.
        device:         torch.device.
        idx_to_class:   Mapping index → tên class.
        is_multimodal:  True nếu dùng MultimodalClassifier + MultimodalTestDataset.
    """

    def __init__(
        self,
        model,
        test_loader,
        device,
        idx_to_class,
        is_multimodal: bool = False,
    ):
        self.model         = model
        self.test_loader   = test_loader
        self.device        = device
        self.idx_to_class  = idx_to_class
        self.is_multimodal = is_multimodal

    def predict(self, model_path=None, output_csv="submission.csv"):
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded model from: {model_path}")

        self.model.eval()
        results = []

        print(f"Running inference on {len(self.test_loader.dataset)} test images...")
        with torch.no_grad():
            for batch in self.test_loader:
                if self.is_multimodal:
                    # MultimodalTestDataset → (hs, ms, rgb, fname)
                    hs, ms, rgb, fnames = batch
                    hs  = hs.to(self.device)
                    ms  = ms.to(self.device)
                    rgb = rgb.to(self.device)
                    outputs = self.model(hs, ms, rgb)
                else:
                    # RGBTestDataset → (rgb, fname)
                    rgb, fnames = batch
                    rgb = rgb.to(self.device)
                    outputs = self.model(rgb)

                preds = outputs.argmax(dim=1).cpu().numpy()

                for fname, p in zip(fnames, preds):
                    # Đổi extension sang .tif cho submission format
                    base = os.path.splitext(fname)[0]
                    results.append({
                        "Id": base + ".tif",
                        "Category": self.idx_to_class[p],
                    })

        df = pd.DataFrame(results)

        out_dir = os.path.dirname(output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")

        return df
