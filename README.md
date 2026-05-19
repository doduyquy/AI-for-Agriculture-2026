<div align="center">

# AI for Agriculture 2026

**Deep Learning Pipeline for Agricultural Image Classification**

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-🔥-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![WandB](https://img.shields.io/badge/Weights_&_Biases-WandB-FFBE00.svg?logo=weightsandbiases&logoColor=white)](https://wandb.ai/)
[![Code Style](https://img.shields.io/badge/Code_Style-Modular-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

---

</div>

## Overview

A PyTorch-based modular pipeline for RGB agricultural image classification. Features include YAML-based configuration and out-of-the-box integration with Weights & Biases (WandB) for experiment tracking.

## Key Features

- **Modular Architecture:** Separation of models, datasets, trainers, and evaluators.
- **YAML Configuration:** Hyperparameters and paths managed via `src/configs/*.yaml`.
- **WandB Integration:** Automated tracking of metrics, learning rates, and checkpoints.
- **End-to-End Pipeline:** Automated workflow from training to inference generation.

---

## Architecture

<details>
<summary><b>View Directory Tree</b></summary>

```text
.
├── src/
│   ├── configs/            # YAML configurations
│   │   ├── dataset.yaml    # Transform & augmentations
│   │   ├── model.yaml      # Model hyperparams
│   │   ├── paths.yaml      # I/O directories
│   │   └── train.yaml      # Training hyperparams
│   ├── models/             # Network architectures (ResNet18)
│   ├── modules/            # Core logic
│   │   ├── dataset.py      # Datasets & Dataloaders
│   │   ├── evaluate.py     # Metrics computation
│   │   ├── inference.py    # Generates submission.csv
│   │   ├── trainer.py      # Training loop
│   │   └── utils.py        # Utilities
│   ├── main.py             # Main execution script
│   └── opts.py             # Command-line argument parser
├── run.sh                  # Bash script for WandB execution
└── README.md               
```

</details>

---

## Setup

### Prerequisites

Python 3.8+ is required.

```bash
pip install torch torchvision pandas numpy scikit-learn wandb pyyaml
```

### Configuration

Modify files in `src/configs/` before running:

| Config File | Purpose | Key Parameters |
| :--- | :--- | :--- |
| `paths.yaml` | Directory paths | `ROOT_DIR`, `TRAIN_RGB_DIR`, `TEST_RGB_DIR`, `CHECKPOINT_DIR` |
| `train.yaml` | Training settings | `LR`, `BATCH_SIZE`, `EPOCHS`, `SEED`, `VAL_SPLIT`, `SPLIT_MANIFEST_PATH` |
| `dataset.yaml` | Data pipeline | Transforms, augmentations |
| `model.yaml` | Architecture | `MODEL_NAME`, `IMG_SIZE`, `NUM_CLASSES` |

### WandB Initialization

Link your Weights & Biases account:

```bash
wandb login
```
*Note: Update `WANDB_ENTITY` in `run.sh`.*

---

## Execution Guide

### Using Bash Script (Recommended)

Handles environment setup and WandB run names automatically.

```bash
chmod +x run.sh
./run.sh
```

### Using Python

Run with or without WandB tracking:

```bash
# With WandB Tracking
python -m src.main --wandb --wandb_project "AI-for-Agriculture" --wandb_entity "your_username" --wandb_run_name "Exp01"

# Local Testing (No Logging)
python -m src.main
```

### Resuming Training

```bash
# Via Bash
RESUME_PATH="checkpoints/your_checkpoint.pth" ./run.sh

# Via Python
python -m src.main --resume "checkpoints/your_checkpoint.pth"
```

---

## Pipeline Flow

1. **Training:** Model learns from the labeled `train/` split and saves weights to `CHECKPOINT_DIR`.
2. **Internal Evaluation:** By default, `SPLIT_MANIFEST_PATH` points to a fixed 80/20 split. The competition `val/` folder is not used for training or validation.
3. **Submission Inference:** Predictions run on the unlabeled competition `val/` split, generating `submission.csv` at the root.
4. **Final Submit Mode:** After choosing a config, set `SPLIT_MANIFEST_PATH: null` and `VAL_SPLIT: 0.0` to train on 100% of labeled `train/` before generating the final submission.

The default training config uses a fixed split manifest at `splits/seed42_val20/split_manifest.csv`, so experiments reuse the same internal train/validation files. Each run also writes `split_manifest.csv` and `split_summary.csv` to the output directory so the active split can be checked.

On Kaggle, the fixed-split training command is:

```bash
DATA_DIR=/kaggle/input/datasets/lhngphc/datasets-split02/seed42_val20 bash run.sh
```

The run writes the submission file to `/kaggle/working/submission.csv`.

To materialize the fixed split into physical folders first:

```bash
python src/tools/create_fixed_split_dataset.py \
  --data_dir /kaggle/input/datasets/nadkli/data-agriculture/dataset \
  --manifest splits/seed42_val20/split_manifest.csv \
  --output_dir /kaggle/working/data_split/seed42_val20 \
  --overwrite
```

Then train from the physical split while keeping the original competition `val/` untouched for submission:

```bash
DATA_DIR=/kaggle/working/data_split/seed42_val20 bash run.sh
```

If you upload the materialized split as a Kaggle Dataset, point `DATA_DIR` to that uploaded dataset folder. It should contain `train/`, `validation/`, and `val/`.

To run a model experiment YAML:

```bash
EXTRA_CONFIG=src/configs/experiments/resnet34_split02.yaml \
DATA_DIR=/kaggle/input/datasets/lhngphc/datasets-split02/seed42_val20 \
bash run.sh
```

To run RGB-only and skip HS/MS completely:

```bash
EXTRA_CONFIG=src/configs/experiments/rgb_resnet18_split02.yaml \
DATA_DIR=/kaggle/input/datasets/lhngphc/datasets-split02/seed42_val20 \
bash run.sh
```

Use `INPUT_MODE: "rgb"` for RGB-only runs, or `INPUT_MODE: "multimodal"` for HS+MS+RGB late fusion.

Model names are registered in `src/models/__init__.py` with `register_rgb_backbone(...)`.
