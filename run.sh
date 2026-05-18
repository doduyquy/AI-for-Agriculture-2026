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

echo "Starting multimodal training pipeline..."

# 1. Run multimodal training with WandB logging
python -m src.main \
    --wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    ${WANDB_API_KEY:+--wandb_api_key "$WANDB_API_KEY"} \
    ${RESUME_PATH:+--resume "$RESUME_PATH"} \
    ${DATA_DIR:+--data_dir "$DATA_DIR"}

# ==========================================
# Other usage examples (Uncomment to use):
# ==========================================

# 2. Run without WandB:
# python -m src.main

# 3. Override data dir (Kaggle):
# DATA_DIR=/kaggle/input/ai-for-agriculture bash run.sh

# 4. Resume training from a specific checkpoint:
# python -m src.main \
#     --wandb \
#     --wandb_project "${WANDB_PROJECT}" \
#     --wandb_entity "${WANDB_ENTITY}" \
#     --wandb_run_name "${WANDB_RUN_NAME}_resumed" \
#     --resume "checkpoints/multimodal_resnet18_imgsize224_batch32_epoch50_lr0.001_last.pth"
