import os
from src.opts import parse_args
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from src.modules.utils import load_config, set_seed, get_filename_crossplatform
from src.modules.dataset import RGBDataset, RGBTestDataset, get_transforms, split_dataset
from src.models.model import build_model
from src.modules.trainer import Trainer
from src.modules.evaluate import Evaluator
from src.modules.inference import Inferencer

def main():
    args = parse_args()
    
    # Load config from YAMLs
    cfg = load_config(args.configs)

    # VAL_RGB_DIR là submission set (không có label) → dùng làm TEST_RGB_DIR
    # Validation trong training → tự split từ TRAIN_RGB_DIR
    if not getattr(cfg, 'TEST_RGB_DIR', None):
        if getattr(cfg, 'VAL_RGB_DIR', None):
            cfg.TEST_RGB_DIR = cfg.VAL_RGB_DIR
    
    # Automatically override paths if --data_dir is provided
    if hasattr(args, 'data_dir') and args.data_dir:
        cfg.DATA_DIR     = args.data_dir
        cfg.TRAIN_DIR    = os.path.join(cfg.DATA_DIR, 'train')
        cfg.TRAIN_RGB_DIR = os.path.join(cfg.TRAIN_DIR, 'RGB')

        val_path  = os.path.join(cfg.DATA_DIR, 'val')
        test_path = os.path.join(cfg.DATA_DIR, 'test')
        if os.path.exists(val_path):
            cfg.VAL_RGB_DIR  = os.path.join(val_path, 'RGB')
            cfg.TEST_RGB_DIR = cfg.VAL_RGB_DIR   # submission set
        if os.path.exists(test_path):
            cfg.TEST_RGB_DIR = os.path.join(test_path, 'RGB')
    
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

    # Validate đường dẫn bắt buộc
    train_rgb = getattr(cfg, 'TRAIN_RGB_DIR', None)
    test_rgb  = getattr(cfg, 'TEST_RGB_DIR',  None)
    if not train_rgb:
        raise AttributeError("Config thiếu 'TRAIN_RGB_DIR'. Kiểm tra lại YAML hoặc --data_dir.")
    if not os.path.exists(train_rgb):
        raise FileNotFoundError(f"Không tìm thấy TRAIN_RGB_DIR: {train_rgb}")
    if test_rgb and not os.path.exists(test_rgb):
        raise FileNotFoundError(f"Không tìm thấy TEST_RGB_DIR: {test_rgb}")

    # 2.1 Split train set → train / val nội bộ (stratified)
    #     val/RGB/ không có label nên không dùng làm labeled val được.
    val_split = getattr(cfg, 'VAL_SPLIT', 0.2)
    train_files, val_files, class_to_idx = split_dataset(
        train_rgb, val_split=val_split, seed=cfg.SEED
    )

    train_ds = RGBDataset(train_rgb, transform=tfm_train,
                          file_list=train_files, class_to_idx=class_to_idx)
    val_ds   = RGBDataset(train_rgb, transform=tfm_val,
                          file_list=val_files,   class_to_idx=class_to_idx)

    # 2.2 Submission test set (val/RGB/ — không có label)
    if test_rgb:
        test_ds = RGBTestDataset(test_rgb, transform=tfm_val)
    else:
        print("[WARN] TEST_RGB_DIR chưa được cấu hình → bỏ qua bước inference.")
        test_ds = None

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0) if test_ds else None

    print(f"[Data] TRAIN_RGB_DIR : {train_rgb}")
    print(f"[Data] TEST_RGB_DIR  : {test_rgb or '(không có)'}")
    print(f"[Data] Val split     : {val_split*100:.0f}%  |  "
          f"train={len(train_ds)}  val={len(val_ds)}  "
          f"test={len(test_ds) if test_ds else 0}")
    print(f"[Data] Classes       : {class_to_idx}")
    
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
    
    # 6. Evaluation (trên val split nội bộ)
    class_names = [train_ds.idx_to_class[i] for i in range(len(class_to_idx))]
    evaluator = Evaluator(model, val_loader, device, class_names)
    y_true, y_pred, report_dict = evaluator.evaluate(model_path=save_path)

    # 7. Inference (trên submission test set — val/RGB/)
    if test_loader is not None:
        inferencer = Inferencer(model, test_loader, device, train_ds.idx_to_class)
        submission_path = os.path.join(cfg.ROOT_DIR, "submission.csv")
        inferencer.predict(model_path=save_path, output_csv=submission_path)
    else:
        submission_path = None
        print("[INFO] Không có test set → bỏ qua inference.")
    
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
        if submission_path and os.path.exists(submission_path):
            artifact = wandb.Artifact('submission', type='dataset')
            artifact.add_file(submission_path)
            wandb.log_artifact(artifact)
        
        # Save best model checkpoint to wandb
        best_model_artifact = wandb.Artifact('best-model', type='model')
        best_model_artifact.add_file(save_path)
        wandb.log_artifact(best_model_artifact)
        
        # Save last model checkpoint to wandb
        last_save_path = save_path.replace('.pth', '_last.pth')
        if os.path.exists(last_save_path):
            last_model_artifact = wandb.Artifact('latest-model', type='model')
            last_model_artifact.add_file(last_save_path)
            wandb.log_artifact(last_model_artifact)
        
        wandb.finish()

if __name__ == "__main__":
    main()
