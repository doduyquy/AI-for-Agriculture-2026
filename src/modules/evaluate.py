import torch
import numpy as np
from sklearn.metrics import classification_report

class Evaluator:
    def __init__(self, model, val_loader, device, class_names):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.class_names = class_names

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
            for images, labels in self.val_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                
                y_true.extend(labels.numpy())
                y_pred.extend(preds)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        print("\\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        report_dict = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        return y_true, y_pred, report_dict
