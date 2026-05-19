#!/bin/bash

# ==========================================
# Run script for AI for Agriculture 2026
# Late Fusion Multimodal (HS + MS + RGB)
# ==========================================

# Variables
WANDB_PROJECT="AI-for-Agriculture"
WANDB_ENTITY="phucga15062005" # Thay bằng username hoặc team name của bạn
WANDB_RUN_NAME="${WANDB_NAME:-MultimodalLateFusion_$(date '+%Y%m%d-%H%M')}"
WANDB_API_KEY="${WANDB_API_KEY:-}" # Kaggle: set bằng os.environ['WANDB_API_KEY'] trước khi chạy !bash run.sh

RESUME_PATH="${RESUME_PATH:-}"
DATA_DIR="${DATA_DIR:-}"   # Tuỳ chọn: override đường dẫn data (dùng trên Kaggle)
SUBMISSION_DATA_DIR="${SUBMISSION_DATA_DIR:-}" # Tuỳ chọn: data gốc chứa val/ không nhãn để submit
EXTRA_CONFIG="${EXTRA_CONFIG:-}" # Tuỳ chọn: YAML experiment override

CONFIG_ARGS=(
    src/configs/paths_kaggle.yaml
    src/configs/model.yaml
    src/configs/train.yaml
    src/configs/dataset.yaml
)

if [ -n "$EXTRA_CONFIG" ]; then
    CONFIG_ARGS+=("$EXTRA_CONFIG")
fi

echo "Starting multimodal training pipeline..."
echo "Configs: ${CONFIG_ARGS[*]}"

# 1. Run multimodal training with WandB logging
python -m src.main \
    --configs "${CONFIG_ARGS[@]}" \
    --wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    ${WANDB_API_KEY:+--wandb_api_key "$WANDB_API_KEY"} \
    ${RESUME_PATH:+--resume "$RESUME_PATH"} \
    ${DATA_DIR:+--data_dir "$DATA_DIR"} \
    ${SUBMISSION_DATA_DIR:+--submission_data_dir "$SUBMISSION_DATA_DIR"}

# ==========================================
# Other usage examples (Uncomment to use):
# ==========================================

# 2. Run without WandB:
# python -m src.main

# 3. Override data dir (Kaggle):
# DATA_DIR=/kaggle/input/datasets/lhngphc/datasets-split02/seed42_val20 bash run.sh

# 3b. Run an experiment YAML:
# EXTRA_CONFIG=src/configs/experiments/resnet34_split02.yaml \
# DATA_DIR=/kaggle/input/datasets/lhngphc/datasets-split02/seed42_val20 bash run.sh

# 4. Resume training from a specific checkpoint:
# python -m src.main \
#     --wandb \
#     --wandb_project "${WANDB_PROJECT}" \
#     --wandb_entity "${WANDB_ENTITY}" \
#     --wandb_run_name "${WANDB_RUN_NAME}_resumed" \
#     --resume "checkpoints/multimodal_resnet18_imgsize224_batch32_epoch50_lr0.001_last.pth"
