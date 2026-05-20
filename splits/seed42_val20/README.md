# Fixed Split: seed42_val20

- Source: `data/train/RGB` filenames only.
- Seed: `42`
- Validation ratio: `0.2` per class.
- Use `split_manifest.csv` for training through `SPLIT_MANIFEST_PATH`.
- This folder stores filenames, not image data. Kaggle still reads images from `DATA_DIR`.
- For final full-data training, set `SPLIT_MANIFEST_PATH: null` and `VAL_SPLIT: 0.0`.
