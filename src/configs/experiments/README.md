# Experiment YAMLs

Run one experiment on Kaggle:

```bash
EXTRA_CONFIG=src/configs/experiments/resnet34_split02.yaml \
DATA_DIR=/kaggle/input/datasets/lhngphc/datasets-split02/seed42_val20 \
bash run.sh
```

The YAML is loaded last, so values here override `model.yaml` and `train.yaml`.

## Model Registration

The current multimodal model is built in `src/models/model.py`.

- `MODEL_NAME` selects the RGB backbone registered in `src/models/__init__.py`.
- Currently supported keys: `resnet18`, `resnet34`, `resnet50`, `resnet101`.
- To add a new RGB backbone, call `register_rgb_backbone(...)`, then use the same key in an experiment YAML.

The HS/MS branches are `HSBranch` and `MSBranch` in the same file. Their output dims are controlled by:

```yaml
MS_OUT_FEATURES: 256
HS_OUT_FEATURES: 256
FUSION_HIDDEN: 512
DROPOUT_P: 0.3
```
