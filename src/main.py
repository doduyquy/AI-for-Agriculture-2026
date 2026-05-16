import os
from src.opts import parse_args
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from src.modules.utils import load_config, set_seed, get_filename_crossplatform
from src.modules.dataset import RGBDataset, RGBTestDataset, get_transforms
from src.models.model import ResNet18Model
from src.modules.trainer import Trainer
from src.modules.evaluate import Evaluator
from src.modules.inference import Inferencer

def main():
    args = parse_args()
    
    # Load config from YAMLs
    cfg = load_config(args.configs)
    
    if args.wandb:
        import wandb
        run_name = args.wandb_run_name if args.wandb_run_name else f"{cfg.MODEL_NAME}_imgsize{cfg.IMG_SIZE}_batch{cfg.BATCH_SIZE}_lr{cfg.LR}"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=dict(cfg))
        wandb.config.update(vars(args))
    
    # 1. Initialization
    set_seed(cfg.SEED)
    device = cfg.device
    print(f"Using device: {device}")
    
    # 2. Data Preparation
    tfm_train, tfm_val = get_transforms(cfg)
    
    # Load splits
    df = pd.read_csv(cfg.SAMPLES_MASTER)
    train_idx = np.load(cfg.TRAIN_IDX_FILE)
    val_idx = np.load(cfg.VAL_IDX_FILE)
    
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    
    train_files = [get_filename_crossplatform(p) for p in df_train["rgb_path"]]
    val_files = [get_filename_crossplatform(p) for p in df_val["rgb_path"]]
    
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
    model = ResNet18Model(num_classes=cfg.NUM_CLASSES, pretrained=True)
    model = model.to(device)
    
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
