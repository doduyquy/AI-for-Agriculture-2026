import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    # Data from user screenshot
    # Rows correspond to True Class: Health, Other, Rust
    # Columns correspond to Predicted Class: Health, Other, Rust
    cm = np.array([
        [31, 2, 6],   # Health
        [4, 31, 2],   # Other
        [12, 1, 27]   # Rust
    ])
    class_names = ["Health", "Other", "Rust"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=160)
    
    # Use standard Blues cmap
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    # Labels and ticks
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    # Draw black grid borders around the cells
    # By setting minor ticks between major ticks and turning on the minor grid
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
            threshold = cm.max() / 2.0
            color = "white" if count > threshold else "black"
            
            ax.text(j, i, cell_text, ha="center", va="center", color=color, fontsize=9, fontweight="normal")

    # Add colorbar
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/final_val_confusion_matrix.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix recreated successfully at {out_path}")

if __name__ == "__main__":
    main()
