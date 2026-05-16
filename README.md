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
| `train.yaml` | Training settings | `LR`, `BATCH_SIZE`, `EPOCHS`, `SEED` |
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

1. **Training:** Model learns and saves weights to `CHECKPOINT_DIR`.
2. **Evaluation:** Best weights evaluate the Validation set. Metrics are logged to WandB.
3. **Inference:** Predictions run on the Test set, generating `submission.csv` at the root.
