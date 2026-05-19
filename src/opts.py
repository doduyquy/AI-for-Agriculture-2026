import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training Baseline Model")
    parser.add_argument(
        "--configs", 
        nargs='+', 
        default=[
            "src/configs/paths_kaggle.yaml", 
            "src/configs/model.yaml", 
            "src/configs/train.yaml", 
            "src/configs/dataset.yaml"
        ], 
        help="Paths to config files"
    )
    parser.add_argument(
        "--wandb", 
        action="store_true", 
        help="Enable wandb logging"
    )
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default="AI-for-Agriculture-2026", 
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_run_name", 
        type=str, 
        default=None, 
        help="Wandb run name"
    )
    parser.add_argument(
        "--wandb_entity", 
        type=str, 
        default=None, 
        help="Wandb entity (username or team name)"
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="Wandb API Key for login"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override base data directory (automatically updates train/test paths)"
    )
    parser.add_argument(
        "--submission_data_dir",
        type=str,
        default=None,
        help="Optional original competition data directory containing unlabeled val/ for submission"
    )
    args = parser.parse_args()
    return args
