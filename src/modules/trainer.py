import os
import time
import torch
import numpy as np

from src.modules.reporting import save_confusion_matrix_artifacts, save_history_csv


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
        output_dir=None,
        class_names=None,
        log_batch_every: int = 0,
        log_confusion_every: int = 0,
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
        self.output_dir    = output_dir
        self.class_names   = class_names
        self.log_batch_every = max(0, int(log_batch_every or 0))
        self.log_confusion_every = max(0, int(log_confusion_every or 0))

        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        self.history_rows = []
        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.best_train_loss = float("inf")
        self.history_csv_path = (
            os.path.join(output_dir, "training_history.csv") if output_dir else None
        )

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

            if self.log_batch_every and (batch_idx % self.log_batch_every == 0 or batch_idx == n_batches):
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
        y_true, y_pred = [], []

        for batch in self.val_loader:
            outputs, labels = self._forward(batch)

            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

        return total_loss / total, correct / total, np.array(y_true), np.array(y_pred)

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
                self.best_val_loss = checkpoint.get('best_val_loss', float("inf"))
                self.best_epoch = checkpoint.get('best_epoch', 0)
                self.best_train_loss = checkpoint.get('best_train_loss', float("inf"))
                if 'history' in checkpoint:
                    self.history = checkpoint['history']
                if 'history_rows' in checkpoint:
                    self.history_rows = checkpoint['history_rows']
                print(
                    f"  ↩  Restored epoch={start_epoch - 1}  "
                    f"best_val_acc={self.best_val_acc:.4f}  "
                    f"best_epoch={self.best_epoch}  "
                    f"best_train_loss={self.best_train_loss:.4f}"
                )
            else:
                self.model.load_state_dict(checkpoint)   # fallback
                print("  ↩  Loaded weights (legacy format — no optimizer/scheduler state)")
            _sep()

        # Ensure checkpoint directory exists
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        # ── Training banner ────────────────────────────────────────────────
        mode_tag = "Multimodal Late Fusion (HS + MS + RGB)" if self.is_multimodal else "RGB-only (ResNet)"
        _header(f"TRAINING START  ·  {mode_tag}")
        print(f"  Epochs     : {start_epoch} → {self.epochs}")
        print(f"  LR (init)  : {self._current_lr():.2e}")
        print(f"  Device     : {self.device}")
        print(f"  Checkpoint : {self.save_path}")
        if self.output_dir:
            print(f"  Output dir : {self.output_dir}")
            print(f"  History CSV: {self.history_csv_path}")
        if self.val_loader is None:
            print("  Validation : disabled (train full labeled set; save best by train loss)")
        else:
            print("  Validation : internal labeled split enabled")
        _sep()
        print(
            "  epoch | lr       | train_loss | train_acc | val_loss | val_acc | best_val | best_ep | time"
        )
        print(
            "  ------+----------+------------+-----------+----------+---------+----------+---------+------"
        )

        total_train_time = 0.0

        for epoch in range(start_epoch, self.epochs + 1):
            epoch_t0 = time.time()
            lr_before = self._current_lr()

            train_loss, train_acc = self.train_one_epoch(epoch)

            # ── Evaluate ───────────────────────────────────────────────────
            val_loss, val_acc = None, None
            y_true, y_pred = None, None
            if self.val_loader is not None:
                val_loss, val_acc, y_true, y_pred = self.evaluate()

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

            # ── Best model ────────────────────────────────────────────────
            if val_acc is not None:
                acc_is_better = val_acc > self.best_val_acc
                acc_is_tied = np.isclose(val_acc, self.best_val_acc)
                loss_breaks_tie = val_loss is not None and val_loss < self.best_val_loss
                is_best = acc_is_better or (acc_is_tied and loss_breaks_tie)
                if is_best:
                    self.best_val_acc = val_acc
                    self.best_val_loss = val_loss if val_loss is not None else float("inf")
                    self.best_epoch = epoch
            else:
                is_best = train_loss < self.best_train_loss
                if is_best:
                    self.best_train_loss = train_loss
                    self.best_epoch = epoch

            epoch_time = time.time() - epoch_t0
            total_train_time += epoch_time
            row = {
                "epoch": epoch,
                "lr": lr_after,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": self.best_val_acc,
                "best_val_loss": self.best_val_loss,
                "best_epoch": self.best_epoch,
                "best_train_loss": self.best_train_loss,
                "epoch_time_sec": epoch_time,
                "is_best": is_best,
            }
            self.history_rows.append(row)
            if self.history_csv_path:
                save_history_csv(self.history_rows, self.history_csv_path)

            should_log_confusion = (
                self.val_loader is not None
                and y_true is not None
                and y_pred is not None
                and self.class_names
                and self.log_confusion_every
                and (epoch % self.log_confusion_every == 0 or epoch == self.epochs or is_best)
            )
            confusion_paths = {}
            if should_log_confusion and self.output_dir:
                confusion_paths = save_confusion_matrix_artifacts(
                    y_true,
                    y_pred,
                    self.class_names,
                    self.output_dir,
                    prefix=f"val_epoch_{epoch:03d}",
                )

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
                    if should_log_confusion:
                        log_payload["val/confusion_matrix"] = wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=y_true,
                            preds=y_pred,
                            class_names=self.class_names,
                        )
                    wandb.log(log_payload)
            except ImportError:
                pass

            # ── Checkpoint ────────────────────────────────────────────────
            checkpoint_state = {
                'epoch'               : epoch,
                'model_state_dict'    : self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                'best_val_acc'        : self.best_val_acc,
                'best_val_loss'       : self.best_val_loss,
                'best_epoch'          : self.best_epoch,
                'best_train_loss'     : self.best_train_loss,
                'history'             : self.history,
                'history_rows'        : self.history_rows,
                'is_multimodal'       : self.is_multimodal,
            }

            last_save_path = self.save_path.replace('.pth', '_last.pth')
            torch.save(checkpoint_state, last_save_path)

            if is_best:
                torch.save(checkpoint_state, self.save_path)

            # ── Epoch summary ─────────────────────────────────────────────
            best_mark = "*" if is_best else ""
            val_loss_text = f"{val_loss:.4f}" if val_loss is not None else "   n/a"
            val_acc_text = f"{val_acc:.4f}" if val_acc is not None else "   n/a"
            best_val_text = f"{self.best_val_acc:.4f}" if self.val_loader is not None else "   n/a"
            print(
                f"  {epoch:>5d} | {lr_after:.2e} | {train_loss:>10.4f} |"
                f" {train_acc:>9.4f} | {val_loss_text:>8} | {val_acc_text:>7} |"
                f" {best_val_text:>8}{best_mark} | {self.best_epoch:>7d} | {epoch_time:>4.0f}s"
            )
            if confusion_paths:
                print(f"        confusion csv: {confusion_paths.get('csv')}")
                if confusion_paths.get("png"):
                    print(f"        confusion png: {confusion_paths.get('png')}")

        # ── Training complete ──────────────────────────────────────────────
        _header("TRAINING COMPLETE")
        print(f"  Total time   : {total_train_time/60:.1f} min")
        if self.val_loader is not None:
            print(f"  Best val_acc : {self.best_val_acc:.4f}")
            print(f"  Best val_loss: {self.best_val_loss:.4f}")
            print(f"  Best epoch   : {self.best_epoch}")
        else:
            print(f"  Best train_loss : {self.best_train_loss:.4f}")
            print(f"  Best epoch      : {self.best_epoch}")
        print(f"  Best model   : {self.save_path}")
        print(f"  Last model   : {self.save_path.replace('.pth', '_last.pth')}")
        _sep()

        return self.history
