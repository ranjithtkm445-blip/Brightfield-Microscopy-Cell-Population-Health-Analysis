"""
instance_detection.py
---------------------
Step 4: Instance Detection

Purpose:
  Load the trained U-Net, run inference on a batch of images,
  and separate the binary mask into individual cell instances
  using connected components.

  Each detected cell gets a unique ID, bounding box, and pixel region.
  This is the bridge between segmentation (Step 3) and
  feature extraction (Step 5).

Input:
  D:/BRIGHT FIELD/models/unet_best.pth     ->  trained U-Net
  D:/BRIGHT FIELD/data/processed/*.npy     ->  preprocessed images

Output:
  D:/BRIGHT FIELD/outputs/instance_results.json  ->  per-image cell counts
  D:/BRIGHT FIELD/outputs/instance_overlay/      ->  overlay images
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import json
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(r"D:\BRIGHT FIELD")
PROC_DIR    = BASE_DIR / "data" / "processed"
MODEL_PATH  = BASE_DIR / "models" / "unet_best.pth"
OUTPUT_DIR  = BASE_DIR / "outputs" / "instance_overlay"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = BASE_DIR / "outputs" / "instance_results.json"


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
    """Run U-Net on single image, return binary mask."""
    tensor = torch.tensor(img[None, None]).to(device)
    with torch.no_grad():
        prob = model(tensor).squeeze().cpu().numpy()
    return (prob > threshold).astype(np.uint8) * 255


# ── Instance separation ───────────────────────────────────────────────────────
def get_instances(binary_mask, min_area=50, max_area=5000):
    """
    Connected components → individual cell instances.
    Filters out noise (too small) and artefacts (too large).
    Returns label_map and count.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean  = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    n_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(clean)

    filtered = np.zeros_like(label_map)
    count    = 0
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            count += 1
            filtered[label_map == i] = count

    return filtered, count


# ── Overlay visualisation ─────────────────────────────────────────────────────
def draw_overlay(img, label_map, n_cells):
    """
    Draw coloured overlay — each cell gets a unique random colour.
    Returns BGR uint8 image.
    """
    rgb = (np.stack([img] * 3, axis=-1) * 255).astype(np.uint8)
    np.random.seed(42)

    for cell_id in range(1, n_cells + 1):
        color = tuple(np.random.randint(50, 255, 3).tolist())
        rgb[label_map == cell_id] = color

    return rgb


# ── Main ──────────────────────────────────────────────────────────────────────
def run_instance_detection(max_images=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")

    # Load model
    model = UNet(base=32).to(device)
    model.load_state_dict(torch.load(MODEL_PATH,
                                     map_location=device,
                                     weights_only=True))
    model.eval()
    print(f"Model      : loaded from {MODEL_PATH}")

    img_paths = sorted(PROC_DIR.glob("*.npy"))[:max_images]
    print(f"Images     : {len(img_paths)}\n")

    results = []

    for img_path in tqdm(img_paths, desc="Instance detection"):
        img        = np.load(img_path).astype(np.float32)
        mask       = predict_mask(model, img, device)
        label_map, n_cells = get_instances(mask)
        confluency = round(float((mask > 0).mean() * 100), 2)

        # Save overlay
        overlay    = draw_overlay(img, label_map, n_cells)
        out_path   = OUTPUT_DIR / (img_path.stem + "_overlay.png")
        cv2.imwrite(str(out_path), overlay)

        results.append({
            "filename"  : img_path.name,
            "n_cells"   : n_cells,
            "confluency": confluency,
        })

    # Save JSON
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    avg_cells = round(np.mean([r["n_cells"]   for r in results]), 1)
    avg_conf  = round(np.mean([r["confluency"] for r in results]), 2)

    print(f"\n✅  Instance detection complete.")
    print(f"    Images processed : {len(results)}")
    print(f"    Avg cells / image: {avg_cells}")
    print(f"    Avg confluency   : {avg_conf}%")
    print(f"    Overlays saved   : {OUTPUT_DIR}")
    print(f"    Results JSON     : {RESULTS_JSON}")


if __name__ == "__main__":
    run_instance_detection(max_images=20)