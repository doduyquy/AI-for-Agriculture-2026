import json
import os
import random

nb_path = r"d:\HocTap\NCKH_ThayDoNhuTai\Challenges\Notebooks\00_baseline\BaselineHS_TopK.ipynb"

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Helper to find cell by content
def find_cell_index(content_snippet):
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if content_snippet in source:
                return i
    return -1

# --- 1. Fix generate_submission (Lỗi 1, 2, 3, 4, 5) ---
# We replace the entire cell containing 'def generate_submission'
gen_sub_idx = find_cell_index("def generate_submission")

new_gen_sub_source = [
    "# --- SUBMISSION GENERATION (FIXED) ---\n",
    "class HSTestDataset(Dataset):\n",
    "    def __init__(self, img_dir, band_indices=None, mean=None, std=None):\n",
    "        self.img_dir = img_dir\n",
    "        self.files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.tif', '.tiff'))])\n",
    "        self.band_indices = band_indices\n",
    "        \n",
    "        # Handle Mean/Std Subset (Global Stats)\n",
    "        if band_indices is not None:\n",
    "            self.mean = torch.tensor(mean[band_indices]).view(-1, 1, 1).float()\n",
    "            self.std = torch.tensor(std[band_indices]).view(-1, 1, 1).float()\n",
    "        else:\n",
    "            self.mean = torch.tensor(mean).view(-1, 1, 1).float()\n",
    "            self.std = torch.tensor(std).view(-1, 1, 1).float()\n",
    "\n",
    "    def __len__(self): return len(self.files)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        fname = self.files[idx]\n",
    "        path = os.path.join(self.img_dir, fname)\n",
    "        arr = tiff.imread(path).astype(np.float32)\n",
    "        \n",
    "        # Robust Dims\n",
    "        if arr.ndim == 3 and arr.shape[-1] >= 120 and arr.shape[-1] <= 130:\n",
    "            arr = np.transpose(arr, (2, 0, 1))\n",
    "        elif arr.ndim == 2: \n",
    "            arr = arr[None, :, :]\n",
    "        \n",
    "        # Force 125\n",
    "        c = arr.shape[0]\n",
    "        if c > 125: arr = arr[:125] \n",
    "        elif c < 125:\n",
    "            pad = np.zeros((125 - c, arr.shape[1], arr.shape[2]), dtype=np.float32)\n",
    "            arr = np.concatenate([arr, pad], axis=0)\n",
    "\n",
    "        # Subset\n",
    "        if self.band_indices is not None:\n",
    "            arr = arr[self.band_indices]\n",
    "            \n",
    "        x = torch.from_numpy(arr)\n",
    "        if x.shape[1:] != TARGET_HW:\n",
    "            x = F.interpolate(x.unsqueeze(0), size=TARGET_HW, mode='bilinear', align_corners=False).squeeze(0)\n",
    "            \n",
    "        x = (x - self.mean) / (self.std + 1e-8)\n",
    "        \n",
    "        # Check #1: Assert shape for top-k\n",
    "        if self.band_indices is not None:\n",
    "             assert x.shape[0] == len(self.band_indices), f\"Band mismatch: {x.shape[0]} vs {len(self.band_indices)}\"\n",
    "        \n",
    "        return x, fname\n",
    "\n",
    "def generate_submission(k, mode, top_idx):\n",
    "    print(f\"\\nGenerating submission for Top-{k} ({mode})...\")\n",
    "    test_dir = r\"D:\\HocTap\\NCKH_ThayDoNhuTai\\Challenges\\data\\raw\\Kaggle_Prepared\\val\\HS\"\n",
    "    ds = HSTestDataset(test_dir, band_indices=top_idx, mean=full_mean, std=full_std)\n",
    "    \n",
    "    # Check #5: Shuffle=False\n",
    "    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)\n",
    "    \n",
    "    # Check #2: Mean/Std shapes\n",
    "    print(f\"Debug: Mean Shape: {ds.mean.shape}, Std Shape: {ds.std.shape}\")\n",
    "    \n",
    "    # Load Model\n",
    "    model = models.resnet18(weights=None)\n",
    "    model.conv1 = nn.Conv2d(k, 64, kernel_size=7, stride=2, padding=3, bias=False)\n",
    "    model.fc = nn.Linear(model.fc.in_features, 3)\n",
    "    \n",
    "    ckpt = os.path.join(CHECKPOINT_DIR, f\"best_top{k}_{mode}.pth\")\n",
    "    print(f\"Loading checkpoint: {ckpt}\")\n",
    "    try:\n",
    "        model.load_state_dict(torch.load(ckpt))\n",
    "    except Exception as e:\n",
    "        print(f\"WARNING: Checkpoint load failed or mismatch! {e}\")\n",
    "        return # Stop if model is wrong\n",
    "        \n",
    "    model.to(device).eval()\n",
    "    \n",
    "    # --- CHECK #3: VALIDATION SANITY CHECK ---\n",
    "    # Run a quick validation to ensure 0.658 compatibility\n",
    "    print(\"Running sanity check on validation set...\")\n",
    "    v_ds = HSFlexibleDataset(HS_DIR, val_files, band_indices=top_idx, augment=False, mean=full_mean, std=full_std)\n",
    "    v_loader = DataLoader(v_ds, batch_size=BATCH_SIZE, shuffle=False)\n",
    "    corr = 0; tot = 0\n",
    "    with torch.no_grad():\n",
    "        for x, y in v_loader:\n",
    "            out = model(x.to(device))\n",
    "            corr += (out.argmax(1) == y.to(device)).sum().item()\n",
    "            tot += x.size(0)\n",
    "    val_acc = corr / tot\n",
    "    print(f\"Sanity Check Val Acc: {val_acc:.4f}\")\n",
    "    if val_acc < 0.55:\n",
    "        print(\"!!! DANGER: Validation accuracy is suspiciously low. Are you using the right Model/Indices?\")\n",
    "    \n",
    "    # Check #1 (Values) - Run one batch\n",
    "    with torch.no_grad():\n",
    "        try:\n",
    "            x_debug, _ = next(iter(loader))\n",
    "            # Print means of first and last channel to check for zeroing/normalization issues\n",
    "            print(f\"Debug Batch 0: x[0,0].mean={x_debug[0,0].mean():.4f}, x[0,{k-1}].mean={x_debug[0,k-1].mean():.4f}\")\n",
    "        except Exception as e: print(f\"Debug batch failed: {e}\")\n",
    "        \n",
    "    preds = []\n",
    "    fnames = []\n",
    "    \n",
    "    with torch.no_grad():\n",
    "        for x, names in loader:\n",
    "            out = model(x.to(device))\n",
    "            preds.extend(out.argmax(1).cpu().numpy())\n",
    "            fnames.extend(names)\n",
    "            \n",
    "    # Map Indices back to Label Names\n",
    "    label_map = {0: 'Health', 1: 'Other', 2: 'Rust'}\n",
    "    \n",
    "    # Check #4: Label Mapping Consistency\n",
    "    all_train_labels = sorted({f.split('_')[0] for f in os.listdir(HS_DIR)})\n",
    "    if len(all_train_labels) == 3:\n",
    "         calc_map = {i: c for i, c in enumerate(all_train_labels)}\n",
    "         if calc_map != label_map:\n",
    "             print(f\"CRITICAL WARNING: Label map mismatch! Train: {calc_map} vs Hardcoded: {label_map}\")\n",
    "             \n",
    "    pred_labels = [label_map[p] for p in preds]\n",
    "    \n",
    "    import pandas as pd\n",
    "    df = pd.DataFrame({'Id': fnames, 'Category': pred_labels})\n",
    "    csv_path = os.path.join(CHECKPOINT_DIR, f\"submission_top{k}_{mode}.csv\")\n",
    "    df.to_csv(csv_path, index=False)\n",
    "    print(f\"Saved {csv_path}\")\n"
]

if gen_sub_idx != -1:
    nb['cells'][gen_sub_idx]['source'] = new_gen_sub_source
else:
    print("Could not find generate_submission cell")

# --- 2. Add Random-30 Baseline (Lỗi 6) ---
# Insert after 'train_model_variant' definition (assuming it's around cell 8) or before the final results loop.
# Let's find the results loop: "for k, res in results.items():"
results_loop_idx = find_cell_index("for k, res in results.items():")

new_random_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- RANDOM-30 BASELINE (Check #6) ---\n",
        "print(\"\\n=== Running Random-30 Baseline ===\")\n",
        "# Select 30 random indices\n",
        "random_idx_30 = sorted(np.random.choice(range(ALL_BANDS), 30, replace=False))\n",
        "print(\"Random 30 Indices:\", random_idx_30)\n",
        "\n",
        "acc_rand = train_model_variant(30, random_idx_30, mode='scratch')\n",
        "print(f\"Random-30 Scratch Accuracy: {acc_rand:.3f}\")\n"
    ]
}

if results_loop_idx != -1:
    # Insert before the results loop
    nb['cells'].insert(results_loop_idx, new_random_cell)
else:
    # Append to end
    nb['cells'].append(new_random_cell)


with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully!")
