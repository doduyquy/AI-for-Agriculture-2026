import os
from src.opts import parse_args
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.modules.utils import load_config, set_seed, get_filename_crossplatform
from src.modules.dataset import RGBDataset, RGBTestDataset, get_transforms
from src.models.model import build_model
from src.modules.trainer import Trainer
from src.modules.evaluate import Evaluator
from src.modules.inference import Inferencer

def main():
    args = parse_args()
    
    # Load config from YAMLs
    cfg = load_config(args.configs)
    
    # Automatically override paths if --data_dir is provided
    if hasattr(args, 'data_dir') and args.data_dir:
        cfg.DATA_DIR = args.data_dir
        cfg.TRAIN_DIR = os.path.join(cfg.DATA_DIR, 'train')
        cfg.TEST_DIR = os.path.join(cfg.DATA_DIR, 'test')
        cfg.TRAIN_RGB_DIR = os.path.join(cfg.TRAIN_DIR, 'RGB')
        cfg.TEST_RGB_DIR = os.path.join(cfg.TEST_DIR, 'RGB')
    
    if args.wandb:
        import wandb
        
        # Check for WANDB_API_KEY in environment or arguments
        wandb_api_key = os.environ.get("WANDB_API_KEY", args.wandb_api_key)
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
            
        run_name = args.wandb_run_name if args.wandb_run_name else f"{cfg.MODEL_NAME}_imgsize{cfg.IMG_SIZE}_batch{cfg.BATCH_SIZE}_lr{cfg.LR}"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=dict(cfg))
        wandb.config.update(vars(args))
    
    # 1. Initialization
    set_seed(cfg.SEED)
    device = cfg.device
    print(f"Using device: {device}")
    
    # 2. Data Preparation
    tfm_train, tfm_val = get_transforms(cfg)
    
    # 2.5 Automatically discover and split dataset
    all_train_files = sorted([f for f in os.listdir(cfg.TRAIN_RGB_DIR) if f.lower().endswith(".png")])
    
    if len(all_train_files) == 0:
        raise FileNotFoundError(f"Không tìm thấy ảnh .png nào trong {cfg.TRAIN_RGB_DIR}")
        
    train_files, val_files = train_test_split(all_train_files, test_size=0.2, random_state=cfg.SEED)
    
    # Create datasets
    train_ds = RGBDataset(cfg.TRAIN_RGB_DIR, transform=tfm_train, file_list=train_files)
    val_ds = RGBDataset(cfg.TRAIN_RGB_DIR, transform=tfm_val, file_list=val_files)
    test_ds = RGBTestDataset(cfg.TEST_RGB_DIR, transform=tfm_val)
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")
    
    # 3. Model Definition
    model = build_model(cfg=cfg, device=device, pretrained=True, dropout_p=0.3)
    
    # 4. Training Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    save_name = f"baseline_rgb_{cfg.MODEL_NAME}_imgsize{cfg.IMG_SIZE}_batch{cfg.BATCH_SIZE}_epoch{cfg.EPOCHS}_lr{cfg.LR}.pth"
    save_path = os.path.join(cfg.CHECKPOINT_DIR, save_name)
    
    # 5. Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_path=save_path,
        epochs=cfg.EPOCHS
    )
    
    history = trainer.train(resume_path=args.resume)
    
    # 6. Evaluation
    class_names = [val_ds.idx_to_class[i] for i in range(cfg.NUM_CLASSES)]
    evaluator = Evaluator(model, val_loader, device, class_names)
    y_true, y_pred, report_dict = evaluator.evaluate(model_path=save_path)
    
    # 7. Inference
    inferencer = Inferencer(model, test_loader, device, val_ds.idx_to_class)
    submission_path = os.path.join(cfg.ROOT_DIR, "submission.csv")
    inferencer.predict(model_path=save_path, output_csv=submission_path)
    
    if args.wandb:
        import wandb
        
        # Log classification report metrics
        for class_name, metrics in report_dict.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    wandb.summary[f"eval/{class_name}/{metric_name}"] = value
            else:
                wandb.summary[f"eval/{class_name}"] = metrics
                
        # Log confusion matrix
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=y_true, preds=y_pred,
                        class_names=class_names)})
        
        # Save submission file as artifact
        artifact = wandb.Artifact('submission', type='dataset')
        artifact.add_file(submission_path)
        wandb.log_artifact(artifact)
        
        # Save best model checkpoint to wandb
        model_artifact = wandb.Artifact('best-model', type='model')
        model_artifact.add_file(save_path)
        wandb.log_artifact(model_artifact)
        
        wandb.finish()

if __name__ == "__main__":
    main()
