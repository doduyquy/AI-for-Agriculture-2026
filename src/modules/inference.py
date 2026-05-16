# pyrefly: ignore [missing-import]
import torch
import pandas as pd
import os

class Inferencer:
    def __init__(self, model, test_loader, device, idx_to_class):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.idx_to_class = idx_to_class

    def predict(self, model_path=None, output_csv="submission.csv"):
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            print(f"Loaded model from: {model_path}")
            
        self.model.eval()
        
        results = []
        
        print(f"Running inference on {len(self.test_loader.dataset)} test images...")
        with torch.no_grad():
            for images, fnames in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                
                for fname, p in zip(fnames, preds):
                    label_name = self.idx_to_class[p]
                    results.append({
                        "filename": fname,
                        "label": label_name
                    })
                    
        df = pd.DataFrame(results)
        
        # Ensure output directory exists if provided in path
        out_dir = os.path.dirname(output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")
        
        return df
