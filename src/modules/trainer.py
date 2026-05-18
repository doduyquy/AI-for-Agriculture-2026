import os
import torch
import numpy as np


class Trainer:
    """
    Universal trainer hỗ trợ cả 2 chế độ:
      - RGB-only   : model.forward(images)            → ResNetClassifier
      - Multimodal : model.forward(hs, ms, rgb)       → MultimodalClassifier

    Chế độ được xác định tự động dựa vào kiểu model (is_multimodal).
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        save_path,
        epochs,
        is_multimodal: bool = False,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = criterion
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.device       = device
        self.save_path    = save_path
        self.epochs       = epochs
        self.is_multimodal = is_multimodal

        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        self.best_val_acc = 0.0

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────
    def _forward(self, batch):
        """
        Thực hiện forward pass phù hợp với kiểu model.

        DataLoader với MultimodalDataset trả về: (hs, ms, rgb, labels)
        DataLoader với RGBDataset             trả về: (rgb, labels)

        Returns:
            outputs (Tensor), labels (Tensor)
        """
        if self.is_multimodal:
            hs, ms, rgb, labels = batch
            hs     = hs.to(self.device)
            ms     = ms.to(self.device)
            rgb    = rgb.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(hs, ms, rgb)
        else:
            # Batch từ MultimodalDataset nhưng chỉ dùng RGB (tương thích ngược)
            if len(batch) == 4:
                _, _, rgb, labels = batch
            else:
                rgb, labels = batch
            rgb    = rgb.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(rgb)

        return outputs, labels

    # ──────────────────────────────────────────────
    # Train / Evaluate
    # ──────────────────────────────────────────────
    def train_one_epoch(self):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in self.train_loader:
            outputs, labels = self._forward(batch)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for batch in self.val_loader:
            outputs, labels = self._forward(batch)

            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        return total_loss / total, correct / total

    # ──────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────
    def train(self, resume_path=None):
        start_epoch = 1
        if resume_path and os.path.exists(resume_path):
            print(f"Resuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
                if 'history' in checkpoint:
                    self.history = checkpoint['history']
            else:
                self.model.load_state_dict(checkpoint)   # fallback

        mode_tag = "multimodal (HS+MS+RGB)" if self.is_multimodal else "RGB-only"
        print(f"[Trainer] Mode : {mode_tag}")
        print(f"[Trainer] Starting training from epoch {start_epoch} to {self.epochs}...")

        # Ensure checkpoint directory exists
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        for epoch in range(start_epoch, self.epochs + 1):
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc     = self.evaluate()

            # Scheduler step
            self.scheduler.step(val_acc)

            # Save history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Log to wandb
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "epoch"      : epoch,
                        "train/loss" : train_loss,
                        "train/acc"  : train_acc,
                        "val/loss"   : val_loss,
                        "val/acc"    : val_acc,
                        "lr"         : self.optimizer.param_groups[0]['lr'],
                    })
            except ImportError:
                pass

            # Update best val_acc
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc

            # Create checkpoint state
            checkpoint_state = {
                'epoch'               : epoch,
                'model_state_dict'    : self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_acc'        : self.best_val_acc,
                'history'             : self.history,
                'is_multimodal'       : self.is_multimodal,
            }

            # Save last model
            last_save_path = self.save_path.replace('.pth', '_last.pth')
            torch.save(checkpoint_state, last_save_path)

            # Save best model
            if is_best:
                torch.save(checkpoint_state, self.save_path)
                print(f"Epoch {epoch:02d} | train_acc={train_acc:.4f}  val_acc={val_acc:.4f} ⭐ BEST")
            else:
                print(f"Epoch {epoch:02d} | train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        print(f"\n✓ Best val_acc : {self.best_val_acc:.4f}")
        print(f"✓ Model saved  : {self.save_path}")
        return self.history
