import os
import time
import torch
import numpy as np


# ──────────────────────────────────────────────
# Helpers in ra console
# ──────────────────────────────────────────────
def _sep(char="─", width=72):
    print(char * width)

def _header(title: str, width=72):
    _sep("═", width)
    print(f"  {title}")
    _sep("═", width)


class Trainer:
    """
    Universal trainer hỗ trợ cả 2 chế độ:
      - RGB-only   : model.forward(images)            → ResNetClassifier
      - Multimodal : model.forward(hs, ms, rgb)       → MultimodalClassifier

    Chế độ được xác định tự động dựa vào kiểu model (is_multimodal).
    Tất cả sự kiện quan trọng (epoch, LR thay đổi, checkpoint, ...) đều được in ra.
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
        self.model         = model
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.criterion     = criterion
        self.optimizer     = optimizer
        self.scheduler     = scheduler
        self.device        = device
        self.save_path     = save_path
        self.epochs        = epochs
        self.is_multimodal = is_multimodal

        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        self.best_val_acc = 0.0
        self.best_train_loss = float("inf")

    # ──────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────
    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

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
    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        n_batches = len(self.train_loader)
        t0 = time.time()

        for batch_idx, batch in enumerate(self.train_loader, 1):
            outputs, labels = self._forward(batch)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            # In tiến trình mỗi 10% số batch (tối đa mỗi 10 batch)
            log_every = max(1, n_batches // 10)
            if batch_idx % log_every == 0 or batch_idx == n_batches:
                cur_loss = total_loss / total
                cur_acc  = correct / total
                elapsed  = time.time() - t0
                print(
                    f"  [Epoch {epoch:02d}] Batch {batch_idx:>4d}/{n_batches}"
                    f"  loss={cur_loss:.4f}  acc={cur_acc:.4f}"
                    f"  ({elapsed:.1f}s)",
                    flush=True,
                )

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self):
        if self.val_loader is None:
            raise RuntimeError("evaluate() được gọi nhưng val_loader=None.")

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
            _sep()
            print(f"  ↩  Resuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
                self.best_train_loss = checkpoint.get('best_train_loss', float("inf"))
                if 'history' in checkpoint:
                    self.history = checkpoint['history']
                print(
                    f"  ↩  Restored epoch={start_epoch - 1}  "
                    f"best_val_acc={self.best_val_acc:.4f}  "
                    f"best_train_loss={self.best_train_loss:.4f}"
                )
            else:
                self.model.load_state_dict(checkpoint)   # fallback
                print("  ↩  Loaded weights (legacy format — no optimizer/scheduler state)")
            _sep()

        # Ensure checkpoint directory exists
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        # ── Training banner ────────────────────────────────────────────────
        mode_tag = "Multimodal Late Fusion (HS + MS + RGB)" if self.is_multimodal else "RGB-only (ResNet)"
        _header(f"TRAINING START  ·  {mode_tag}")
        print(f"  Epochs     : {start_epoch} → {self.epochs}")
        print(f"  LR (init)  : {self._current_lr():.2e}")
        print(f"  Device     : {self.device}")
        print(f"  Checkpoint : {self.save_path}")
        if self.val_loader is None:
            print("  Validation : disabled (train full labeled set; save best by train loss)")
        else:
            print("  Validation : internal labeled split enabled")
        _sep()

        total_train_time = 0.0

        for epoch in range(start_epoch, self.epochs + 1):
            epoch_t0 = time.time()
            lr_before = self._current_lr()

            # ── Train ──────────────────────────────────────────────────────
            print(f"\n{'─'*72}")
            print(f"  Epoch {epoch:02d}/{self.epochs}  |  LR = {lr_before:.2e}")
            print(f"{'─'*72}")
            train_loss, train_acc = self.train_one_epoch(epoch)

            # ── Evaluate ───────────────────────────────────────────────────
            val_loss, val_acc = None, None
            if self.val_loader is not None:
                print(f"  → Evaluating on val set...", end=" ", flush=True)
                val_loss, val_acc = self.evaluate()
                print(f"done.")

            # ── Scheduler step (detect LR change) ─────────────────────────
            if self.scheduler is not None and val_acc is not None:
                self.scheduler.step(val_acc)
            lr_after = self._current_lr()
            if lr_after != lr_before:
                print(
                    f"  ⚙  LR Scheduler: {lr_before:.2e} → {lr_after:.2e}"
                    f"  (ReduceLROnPlateau triggered)"
                )

            # ── Save history ───────────────────────────────────────────────
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # ── Log to WandB ───────────────────────────────────────────────
            try:
                import wandb
                if wandb.run is not None:
                    log_payload = {
                        "epoch"      : epoch,
                        "train/loss" : train_loss,
                        "train/acc"  : train_acc,
                        "lr"         : lr_after,
                    }
                    if val_loss is not None and val_acc is not None:
                        log_payload.update({
                            "val/loss": val_loss,
                            "val/acc": val_acc,
                        })
                    wandb.log(log_payload)
            except ImportError:
                pass

            # ── Best model ────────────────────────────────────────────────
            if val_acc is not None:
                is_best = val_acc >= self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
            else:
                is_best = train_loss < self.best_train_loss
                if is_best:
                    self.best_train_loss = train_loss

            # ── Checkpoint ────────────────────────────────────────────────
            checkpoint_state = {
                'epoch'               : epoch,
                'model_state_dict'    : self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                'best_val_acc'        : self.best_val_acc,
                'best_train_loss'     : self.best_train_loss,
                'history'             : self.history,
                'is_multimodal'       : self.is_multimodal,
            }

            last_save_path = self.save_path.replace('.pth', '_last.pth')
            torch.save(checkpoint_state, last_save_path)
            print(f"  💾 Checkpoint saved → {os.path.basename(last_save_path)}")

            if is_best:
                torch.save(checkpoint_state, self.save_path)
                print(f"  💾 Best model saved → {os.path.basename(self.save_path)}")

            # ── Epoch summary ─────────────────────────────────────────────
            epoch_time = time.time() - epoch_t0
            total_train_time += epoch_time

            best_tag = " ⭐ NEW BEST" if is_best else ""
            if val_acc is not None:
                print(
                    f"\n  [Epoch {epoch:02d} Summary] Time: {epoch_time:.1f}s\n"
                    f"   - Train : loss={train_loss:.4f}, acc={train_acc:.4f}\n"
                    f"   - Val   : loss={val_loss:.4f}, acc={val_acc:.4f}{best_tag} (best: {self.best_val_acc:.4f})\n"
                )
            else:
                print(
                    f"\n  [Epoch {epoch:02d} Summary] Time: {epoch_time:.1f}s\n"
                    f"   - Train : loss={train_loss:.4f}, acc={train_acc:.4f}{best_tag} "
                    f"(best loss: {self.best_train_loss:.4f})\n"
                )

        # ── Training complete ──────────────────────────────────────────────
        _header("TRAINING COMPLETE")
        print(f"  Total time   : {total_train_time/60:.1f} min")
        if self.val_loader is not None:
            print(f"  Best val_acc : {self.best_val_acc:.4f}")
        else:
            print(f"  Best train_loss : {self.best_train_loss:.4f}")
        print(f"  Best model   : {self.save_path}")
        print(f"  Last model   : {self.save_path.replace('.pth', '_last.pth')}")
        _sep()

        return self.history
