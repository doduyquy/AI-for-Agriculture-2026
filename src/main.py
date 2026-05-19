import os
from src.opts import parse_args
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from src.modules.utils import load_config, set_seed, get_filename_crossplatform, label_from_filename
from src.modules.dataset import (
    MultimodalDataset,
    MultimodalTestDataset,
    get_transforms,
    load_split_manifest,
    print_split_summary,
    save_split_audit,
    split_dataset,
)
from src.models.model import build_multimodal_model
from src.modules.trainer import Trainer
from src.modules.evaluate import Evaluator
from src.modules.inference import Inferencer


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def list_image_files(img_dir):
    return sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])

def main():
    args = parse_args()
    
    # Load config from YAMLs
    cfg = load_config(args.configs)

    # VAL_RGB_DIR là submission set (không có label) → dùng làm TEST_RGB_DIR.
    # Nếu cần validation nội bộ thì bật VAL_SPLIT > 0 trong train.yaml.
    if not getattr(cfg, 'TEST_RGB_DIR', None):
        if getattr(cfg, 'VAL_RGB_DIR', None):
            cfg.TEST_RGB_DIR = cfg.VAL_RGB_DIR
    
    # Automatically override paths if --data_dir is provided
    if hasattr(args, 'data_dir') and args.data_dir:
        cfg.DATA_DIR      = args.data_dir
        cfg.TRAIN_DIR     = os.path.join(cfg.DATA_DIR, 'train')
        cfg.TRAIN_RGB_DIR = os.path.join(cfg.TRAIN_DIR, 'RGB')
        cfg.TRAIN_HS_DIR  = os.path.join(cfg.TRAIN_DIR, 'HS')
        cfg.TRAIN_MS_DIR  = os.path.join(cfg.TRAIN_DIR, 'MS')

        validation_path = os.path.join(cfg.DATA_DIR, 'validation')
        if os.path.exists(validation_path):
            cfg.VALIDATION_RGB_DIR = os.path.join(validation_path, 'RGB')
            cfg.VALIDATION_HS_DIR  = os.path.join(validation_path, 'HS')
            cfg.VALIDATION_MS_DIR  = os.path.join(validation_path, 'MS')

        val_path = os.path.join(cfg.DATA_DIR, 'val')
        test_path = os.path.join(cfg.DATA_DIR, 'test')
        if os.path.exists(val_path):
            cfg.VAL_RGB_DIR  = os.path.join(val_path, 'RGB')
            cfg.VAL_HS_DIR   = os.path.join(val_path, 'HS')
            cfg.VAL_MS_DIR   = os.path.join(val_path, 'MS')
            cfg.TEST_RGB_DIR = cfg.VAL_RGB_DIR   # submission set
            cfg.TEST_HS_DIR  = cfg.VAL_HS_DIR
            cfg.TEST_MS_DIR  = cfg.VAL_MS_DIR
        if os.path.exists(test_path):
            cfg.TEST_RGB_DIR = os.path.join(test_path, 'RGB')
            cfg.TEST_HS_DIR  = os.path.join(test_path, 'HS')
            cfg.TEST_MS_DIR  = os.path.join(test_path, 'MS')

    if hasattr(args, 'submission_data_dir') and args.submission_data_dir:
        submission_val_path = os.path.join(args.submission_data_dir, 'val')
        if not os.path.exists(submission_val_path):
            raise FileNotFoundError(
                f"Không tìm thấy submission val dir: {submission_val_path}"
            )
        cfg.TEST_RGB_DIR = os.path.join(submission_val_path, 'RGB')
        cfg.TEST_HS_DIR  = os.path.join(submission_val_path, 'HS')
        cfg.TEST_MS_DIR  = os.path.join(submission_val_path, 'MS')
    
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

    # 2.1 Train labeled set.
    #     validation/: dùng folder validation vật lý nếu DATA_DIR đã được tách sẵn.
    #     SPLIT_MANIFEST_PATH: dùng split cố định đã tạo sẵn để công bằng giữa các lần thử.
    #     VAL_SPLIT=0.2: tách 20% từ train/ nếu không có manifest.
    #     VAL_SPLIT=0.0: dùng toàn bộ train/ cho final submit nếu không có manifest.
    #     val/RGB/ của competition không có label nên chỉ dùng cho submission.
    val_split = getattr(cfg, 'VAL_SPLIT', 0.2)
    split_manifest_cfg = getattr(cfg, 'SPLIT_MANIFEST_PATH', None)
    validation_rgb = getattr(cfg, 'VALIDATION_RGB_DIR', None)
    use_physical_validation = validation_rgb and os.path.exists(validation_rgb)
    if use_physical_validation:
        train_files = list_image_files(train_rgb)
        val_files = list_image_files(validation_rgb)
        all_labels = sorted({
            label_from_filename(f)
            for f in train_files + val_files
        })
        class_to_idx = {c: i for i, c in enumerate(all_labels)}
        print(f"[Split] Source        : physical folders ({cfg.TRAIN_DIR} + {os.path.dirname(validation_rgb)})")
    elif split_manifest_cfg:
        split_manifest_cfg = os.path.abspath(split_manifest_cfg)
        if not os.path.exists(split_manifest_cfg):
            raise FileNotFoundError(f"Không tìm thấy SPLIT_MANIFEST_PATH: {split_manifest_cfg}")
        train_files, val_files, class_to_idx = load_split_manifest(
            split_manifest_cfg, img_dir=train_rgb
        )
        print(f"[Split] Source        : fixed manifest ({split_manifest_cfg})")
    else:
        train_files, val_files, class_to_idx = split_dataset(
            train_rgb, val_split=val_split, seed=cfg.SEED
        )
        print(f"[Split] Source        : generated by VAL_SPLIT={val_split} SEED={cfg.SEED}")
    print_split_summary(train_files, val_files, class_to_idx)
    split_audit_dir = getattr(cfg, 'ROOT_DIR', getattr(cfg, 'CHECKPOINT_DIR', '.'))
    split_manifest_path, split_summary_path = save_split_audit(
        split_audit_dir, train_files, val_files, class_to_idx
    )
    print(f"[Split] Manifest saved : {split_manifest_path}")
    print(f"[Split] Summary saved  : {split_summary_path}")

    train_hs = getattr(cfg, 'TRAIN_HS_DIR', os.path.join(cfg.TRAIN_DIR, 'HS') if hasattr(cfg, 'TRAIN_DIR') else train_rgb.replace('RGB', 'HS'))
    train_ms = getattr(cfg, 'TRAIN_MS_DIR', os.path.join(cfg.TRAIN_DIR, 'MS') if hasattr(cfg, 'TRAIN_DIR') else train_rgb.replace('RGB', 'MS'))
    validation_hs = getattr(cfg, 'VALIDATION_HS_DIR', None)
    validation_ms = getattr(cfg, 'VALIDATION_MS_DIR', None)
    if use_physical_validation:
        for name, path in {
            "VALIDATION_HS_DIR": validation_hs,
            "VALIDATION_MS_DIR": validation_ms,
        }.items():
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"Không tìm thấy {name}: {path}")

    train_ds = MultimodalDataset(train_hs, train_ms, train_rgb, transform=tfm_train,
                                 file_list=train_files, class_to_idx=class_to_idx)
    val_ds = None
    if val_files:
        val_rgb_dir = validation_rgb if use_physical_validation else train_rgb
        val_hs_dir = validation_hs if use_physical_validation else train_hs
        val_ms_dir = validation_ms if use_physical_validation else train_ms
        val_ds = MultimodalDataset(val_hs_dir, val_ms_dir, val_rgb_dir, transform=tfm_val,
                                   file_list=val_files, class_to_idx=class_to_idx)

    # 2.2 Submission test set (val/{RGB,HS,MS}/ — đủ cả 3 modality, không có label)
    if test_rgb and os.path.exists(test_rgb):
        test_hs = getattr(cfg, 'TEST_HS_DIR', test_rgb.replace('RGB', 'HS'))
        test_ms = getattr(cfg, 'TEST_MS_DIR', test_rgb.replace('RGB', 'MS'))
        test_ds = MultimodalTestDataset(test_hs, test_ms, test_rgb, transform=tfm_val)
    else:
        print("[WARN] TEST_RGB_DIR chưa được cấu hình → bỏ qua bước inference.")
        test_ds = None

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0) if val_ds else None
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0) if test_ds else None

    print(f"[Data] TRAIN_RGB_DIR : {train_rgb}")
    print(f"[Data] TEST_RGB_DIR  : {test_rgb or '(không có)'}")
    print(f"[Data] Val split     : {val_split*100:.0f}%  |  "
          f"train={len(train_ds)}  val={len(val_ds) if val_ds else 0}  "
          f"test={len(test_ds) if test_ds else 0}")
    print(f"[Data] Classes       : {class_to_idx}")
    
    # 3. Model Definition
    model = build_multimodal_model(cfg=cfg, device=device)

    # 4. Training Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    scheduler = None
    if val_loader is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    save_name = f"multimodal_{cfg.MODEL_NAME}_imgsize{cfg.IMG_SIZE}_batch{cfg.BATCH_SIZE}_epoch{cfg.EPOCHS}_lr{cfg.LR}.pth"
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
        epochs=cfg.EPOCHS,
        is_multimodal=True,
    )
    
    history = trainer.train(resume_path=args.resume)
    
    # 6. Evaluation (chỉ chạy khi có val split nội bộ)
    class_names = [train_ds.idx_to_class[i] for i in range(len(class_to_idx))]
    y_true, y_pred, report_dict = None, None, {}
    if val_loader is not None:
        evaluator = Evaluator(model, val_loader, device, class_names, is_multimodal=True)
        y_true, y_pred, report_dict = evaluator.evaluate(model_path=save_path)
    else:
        print("[INFO] Không có labeled validation split → bỏ qua classification report.")

    # 7. Inference (trên submission set — val/{RGB,HS,MS}/ không nhãn)
    if test_loader is not None:
        inferencer = Inferencer(model, test_loader, device, train_ds.idx_to_class, is_multimodal=True)
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
                
        # Log confusion matrix only when an internal labeled val split exists.
        if y_true is not None and y_pred is not None:
            wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            y_true=y_true, preds=y_pred,
                            class_names=class_names)})
        
        # Save submission file as artifact
        if submission_path and os.path.exists(submission_path):
            artifact = wandb.Artifact('submission', type='dataset')
            artifact.add_file(submission_path)
            wandb.log_artifact(artifact)

        # Save split audit files as artifacts.
        split_artifact = wandb.Artifact('split-audit', type='dataset')
        for path in (split_manifest_path, split_summary_path):
            if os.path.exists(path):
                split_artifact.add_file(path)
        wandb.log_artifact(split_artifact)
        
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
