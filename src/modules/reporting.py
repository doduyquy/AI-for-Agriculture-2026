import csv
import json
import os
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
from sklearn.metrics import confusion_matrix


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_history_csv(rows: Iterable[Mapping], output_path: str) -> str:
    """Save per-epoch training metrics so Kaggle Output keeps a compact run log."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "epoch",
        "lr",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "best_val_acc",
        "best_val_loss",
        "best_epoch",
        "best_train_loss",
        "epoch_time_sec",
        "is_best",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return output_path


def save_confusion_matrix_artifacts(
    y_true,
    y_pred,
    class_names: List[str],
    output_dir: str,
    prefix: str,
) -> Dict[str, Optional[str]]:
    """Save confusion matrix as CSV and PNG under output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(cm.sum())
    correct = int(np.trace(cm))
    accuracy = correct / total if total else 0.0

    csv_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\pred", *class_names])
        for class_name, row in zip(class_names, cm):
            writer.writerow([class_name, *row.tolist()])

    summary_path = os.path.join(output_dir, f"{prefix}_metrics_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["accuracy", accuracy])
        writer.writerow(["accuracy_percent", accuracy * 100.0])
        writer.writerow(["correct", correct])
        writer.writerow(["total", total])

    png_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.png")
    saved_png = _save_confusion_matrix_png(cm, class_names, png_path)

    return {"csv": csv_path, "png": saved_png, "summary": summary_path}


def _save_confusion_matrix_png(cm: np.ndarray, class_names: List[str], output_path: str) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=160)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Labels and ticks
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    # Draw black grid borders around the cells
    ax.set_xticks(np.arange(len(class_names) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(class_names) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Label axes
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_title("Confusion Matrix (Count & Percentage)", fontsize=12, fontweight="bold", pad=12)

    # Row-wise sum for percentage calculation
    row_sums = cm.sum(axis=1)

    # Annotate cells with counts and percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = (count / row_sums[i] * 100.0) if row_sums[i] > 0 else 0.0
            cell_text = f"{count}\n({pct:.2f}%)"
            
            # Decide text color based on cell brightness
            threshold = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0.0
            color = "white" if count > threshold else "black"
            
            ax.text(j, i, cell_text, ha="center", va="center", color=color, fontsize=9, fontweight="normal")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_classification_report_artifacts(
    report_dict: Mapping,
    output_dir: str,
    prefix: str = "final_val",
) -> Dict[str, str]:
    """Save sklearn classification_report output as JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"{prefix}_classification_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False, default=_json_default)

    csv_path = os.path.join(output_dir, f"{prefix}_classification_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["label", "precision", "recall", "f1-score", "support", "value"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, metrics in report_dict.items():
            if isinstance(metrics, Mapping):
                writer.writerow({
                    "label": label,
                    "precision": metrics.get("precision", ""),
                    "recall": metrics.get("recall", ""),
                    "f1-score": metrics.get("f1-score", ""),
                    "support": metrics.get("support", ""),
                    "value": "",
                })
            else:
                writer.writerow({
                    "label": label,
                    "precision": "",
                    "recall": "",
                    "f1-score": "",
                    "support": "",
                    "value": metrics,
                })

    return {"json": json_path, "csv": csv_path}


def log_experiment_to_csv(
    csv_path: str,
    cfg,
    metrics: dict,
    confusion_matrix_img: str = ""
) -> str:
    """Append a row of experiment details and results to a CSV log file."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    
    # Headers match experiments.csv
    fieldnames = [
        "Confusion Matrix",
        "Kaggle score (Public)",
        "Kaggle score (Private)",
        "Val Acc",
        "Val F1",
        "Modality",
        "Model",
        "Input Channels",
        "Image Size",
        "Optimizer",
        "Lr first",
        "Lr after",
        "Lr scheduler",
        "Best Ep",
        "Total Eps"
    ]
    
    file_exists = os.path.exists(csv_path)
    
    # Determine modality string
    input_mode = str(getattr(cfg, "INPUT_MODE", "rgb")).lower().replace("-", "_")
    if input_mode in {"rgb", "rgb_only", "single_rgb"}:
        modality = "RGB"
        input_channels = "3"
    else:
        # Multimodal can include combinations of RGB, MS (5 ch), and HS (125 ch)
        parts = ["RGB"]
        ch_parts = ["3"]
        if getattr(cfg, "TRAIN_MS_DIR", None) or getattr(cfg, "VAL_MS_DIR", None):
            parts.append("MS")
            ch_parts.append("5")
        if getattr(cfg, "TRAIN_HS_DIR", None) or getattr(cfg, "VAL_HS_DIR", None):
            parts.append("HS")
            ch_parts.append("125")
        modality = " + ".join(parts)
        input_channels = " | ".join(ch_parts)
        
    model_name = getattr(cfg, "MODEL_NAME", "unknown")
    img_size = getattr(cfg, "IMG_SIZE", "unknown")
    if isinstance(img_size, int):
        img_size = f"{img_size}x{img_size}"
        
    optimizer = getattr(cfg, "OPTIMIZER", "Adam")
    lr_scheduler = getattr(cfg, "SCHEDULER", "None")
    total_eps = getattr(cfg, "EPOCHS", 0)
    
    lr_first = metrics.get("lr_first", getattr(cfg, "LR", 0.0))
    lr_after = metrics.get("lr_after", lr_first)
    
    val_acc = metrics.get("val_acc", "")
    val_f1 = metrics.get("val_f1", "")
    best_ep = metrics.get("best_epoch", "")
    
    if isinstance(val_acc, float):
        val_acc = f"{val_acc:.4f}"
    if isinstance(val_f1, float):
        val_f1 = f"{val_f1:.4f}"
    if isinstance(lr_first, float):
        lr_first = f"{lr_first:.2e}"
    if isinstance(lr_after, float):
        lr_after = f"{lr_after:.2e}"
        
    row = {
        "Confusion Matrix": confusion_matrix_img,
        "Kaggle score (Public)": " - ",
        "Kaggle score (Private)": " - ",
        "Val Acc": val_acc,
        "Val F1": val_f1,
        "Modality": modality,
        "Model": model_name,
        "Input Channels": input_channels,
        "Image Size": img_size,
        "Optimizer": optimizer,
        "Lr first": lr_first,
        "Lr after": lr_after,
        "Lr scheduler": lr_scheduler,
        "Best Ep": best_ep,
        "Total Eps": total_eps
    }
    
    mode = "a" if file_exists else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        
    return csv_path

