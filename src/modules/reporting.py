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

    csv_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\pred", *class_names])
        for class_name, row in zip(class_names, cm):
            writer.writerow([class_name, *row.tolist()])

    png_path = os.path.join(output_dir, f"{prefix}_confusion_matrix.png")
    saved_png = _save_confusion_matrix_png(cm, class_names, png_path)

    return {"csv": csv_path, "png": saved_png}


def _save_confusion_matrix_png(cm: np.ndarray, class_names: List[str], output_path: str) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Actual",
        xlabel="Predicted",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

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
