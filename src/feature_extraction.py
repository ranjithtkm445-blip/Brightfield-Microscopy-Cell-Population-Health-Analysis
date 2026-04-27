"""
feature_extraction.py
---------------------
Step 5: Feature Extraction

Purpose:
  For each detected cell instance, extract morphological and intensity
  features using skimage.measure.regionprops.

  Features extracted per cell:
    - area          : cell size in pixels
    - perimeter     : cell boundary length
    - circularity   : 4π·area / perimeter² (1=perfect circle)
    - eccentricity  : elongation (0=circle, 1=line)
    - solidity      : area / convex hull area (membrane integrity)
    - mean_intensity: average pixel brightness inside cell
    - std_intensity : pixel brightness variation inside cell

  Population-level metrics per image:
    - n_cells       : total detected cells
    - confluency    : % image area covered by cells
    - mean of each feature across all cells

Input:
  D:/BRIGHT FIELD/models/unet_best.pth     ->  trained U-Net
  D:/BRIGHT FIELD/data/processed/*.npy     ->  preprocessed images

Output:
  D:/BRIGHT FIELD/outputs/features.csv    ->  per-cell features
  D:/BRIGHT FIELD/outputs/population.csv  ->  per-image summary
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import pandas as pd
from tqdm import tqdm
from skimage import measure

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(r"D:\BRIGHT FIELD")
PROC_DIR     = BASE_DIR / "data" / "processed"
MODEL_PATH   = BASE_DIR / "models" / "unet_best.pth"
OUTPUT_DIR   = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_CSV   = OUTPUT_DIR / "features.csv"
POPULATION_CSV = OUTPUT_DIR / "population.csv"


# ── U-Net (same architecture as train.py) ─────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class UNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.inc   = DoubleConv(1,        base)
        self.down1 = Down(base,           base*2)
        self.down2 = Down(base*2,         base*4)
        self.down3 = Down(base*4,         base*8)
        self.down4 = Down(base*8,         base*16)
        self.up1   = Up(base*16 + base*8, base*8)
        self.up2   = Up(base*8  + base*4, base*4)
        self.up3   = Up(base*4  + base*2, base*2)
        self.up4   = Up(base*2  + base,   base)
        self.out   = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return torch.sigmoid(self.out(x))


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_mask(model, img, device, threshold=0.5):
    tensor = torch.tensor(img[None, None]).to(device)
    with torch.no_grad():
        prob = model(tensor).squeeze().cpu().numpy()
    return (prob > threshold).astype(np.uint8)


# ── Instance separation ───────────────────────────────────────────────────────
def get_label_map(binary_mask, min_area=50, max_area=5000):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean  = cv2.morphologyEx(
        (binary_mask * 255).astype(np.uint8),
        cv2.MORPH_OPEN, kernel
    )
    label_map = measure.label(clean, connectivity=2)
    props     = measure.regionprops(label_map)
    filtered  = np.zeros_like(label_map)
    new_id    = 1
    for p in props:
        if min_area <= p.area <= max_area:
            filtered[label_map == p.label] = new_id
            new_id += 1
    return filtered


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(img, label_map):
    """
    Extract per-cell morphological + intensity features
    using skimage regionprops.
    """
    props    = measure.regionprops(label_map, intensity_image=img)
    cells    = []

    for p in props:
        area       = float(p.area)
        perimeter  = float(p.perimeter) if p.perimeter > 0 else 1.0
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
        region_px  = img[label_map == p.label]

        cells.append({
            "cell_id"       : int(p.label),
            "area"          : round(area, 2),
            "perimeter"     : round(perimeter, 2),
            "circularity"   : round(circularity, 4),
            "eccentricity"  : round(float(p.eccentricity), 4),
            "solidity"      : round(float(p.solidity), 4),
            "mean_intensity": round(float(region_px.mean()), 4),
            "std_intensity" : round(float(region_px.std()), 4),
        })

    return cells


# ── Main ──────────────────────────────────────────────────────────────────────
def run_feature_extraction(max_images=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    # Load model
    model = UNet(base=32).to(device)
    model.load_state_dict(torch.load(MODEL_PATH,
                                     map_location=device,
                                     weights_only=True))
    model.eval()
    print(f"Model  : loaded")

    img_paths = sorted(PROC_DIR.glob("*.npy"))
    if max_images:
        img_paths = img_paths[:max_images]
    print(f"Images : {len(img_paths)}\n")

    all_cells  = []
    population = []

    for img_path in tqdm(img_paths, desc="Feature extraction"):
        img       = np.load(img_path).astype(np.float32)
        mask      = predict_mask(model, img, device)
        label_map = get_label_map(mask)
        cells     = extract_features(img, label_map)
        n_cells   = len(cells)
        confluency = round(float((mask > 0).mean() * 100), 2)

        # Tag each cell with its source image
        for c in cells:
            c["filename"] = img_path.name
        all_cells.extend(cells)

        # Population summary
        if n_cells > 0:
            pop = {
                "filename"        : img_path.name,
                "n_cells"         : n_cells,
                "confluency_pct"  : confluency,
                "mean_area"       : round(np.mean([c["area"]           for c in cells]), 2),
                "mean_circularity": round(np.mean([c["circularity"]     for c in cells]), 4),
                "mean_eccentricity":round(np.mean([c["eccentricity"]    for c in cells]), 4),
                "mean_solidity"   : round(np.mean([c["solidity"]        for c in cells]), 4),
                "mean_intensity"  : round(np.mean([c["mean_intensity"]  for c in cells]), 4),
            }
        else:
            pop = {
                "filename": img_path.name,
                "n_cells": 0, "confluency_pct": confluency,
                "mean_area": 0, "mean_circularity": 0,
                "mean_eccentricity": 0, "mean_solidity": 0,
                "mean_intensity": 0,
            }
        population.append(pop)

    # Save CSVs
    pd.DataFrame(all_cells).to_csv(FEATURES_CSV,   index=False)
    pd.DataFrame(population).to_csv(POPULATION_CSV, index=False)

    # Summary
    total_cells = len(all_cells)
    avg_cells   = round(np.mean([p["n_cells"]        for p in population]), 1)
    avg_conf    = round(np.mean([p["confluency_pct"]  for p in population]), 2)
    avg_circ    = round(np.mean([p["mean_circularity"]for p in population]), 4)
    avg_sol     = round(np.mean([p["mean_solidity"]   for p in population]), 4)

    print(f"\n✅  Feature extraction complete.")
    print(f"    Images processed  : {len(population)}")
    print(f"    Total cells       : {total_cells}")
    print(f"    Avg cells / image : {avg_cells}")
    print(f"    Avg confluency    : {avg_conf}%")
    print(f"    Avg circularity   : {avg_circ}")
    print(f"    Avg solidity      : {avg_sol}")
    print(f"    Features CSV      : {FEATURES_CSV}")
    print(f"    Population CSV    : {POPULATION_CSV}")


if __name__ == "__main__":
    run_feature_extraction(max_images=None)