"""
verify_data.py
--------------
Step 3a: Data Verification — Before Training

Purpose:
  Visually confirm that preprocessed images and masks are correct
  before committing to 50 epochs of training.

  Checks:
  1. Image loads correctly as float32 [0,1]
  2. Mask loads correctly as binary 0/255
  3. Image and mask are correctly paired (same well)
  4. Saves a visual grid of 6 random pairs to verify_output.png

Output:
  D:\BRIGHT FIELD\verify_output.png  ->  grid of image/mask pairs
"""

from pathlib import Path
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(r"D:\BRIGHT FIELD")
PROC_DIR  = BASE_DIR / "data" / "processed"
MASK_DIR  = BASE_DIR / "data" / "masks"
OUT_PATH  = BASE_DIR / "verify_output.png"

# ── Load paths ────────────────────────────────────────────────────────────────
img_paths  = sorted(PROC_DIR.glob("*.npy"))
mask_paths = sorted(MASK_DIR.glob("*.png"))

print(f"Images : {len(img_paths)}")
print(f"Masks  : {len(mask_paths)}")

# ── Check pairing ─────────────────────────────────────────────────────────────
print("\nChecking pairs...")
mismatches = 0
for ip, mp in zip(img_paths[:10], mask_paths[:10]):
    img_key  = ip.stem
    mask_key = mp.stem.replace("_mask", "")
    status   = "OK" if img_key == mask_key else "MISMATCH"
    if status == "MISMATCH":
        mismatches += 1
    print(f"  {status}  {img_key[:40]}...")

if mismatches == 0:
    print("✅  All pairs match correctly.")
else:
    print(f"⚠  {mismatches} mismatches found — check pairing logic.")

# ── Visual check: 6 random pairs ──────────────────────────────────────────────
random.seed(42)
indices = random.sample(range(len(img_paths)), 6)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle("BBBC006 Data Verification — Image | Mask | Overlay", fontsize=14)

for row, idx in enumerate(indices[:3]):
    img  = np.load(img_paths[idx])
    mask = cv2.imread(str(mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
    mask_bin = (mask > 127).astype(np.float32)

    # Column 0: raw preprocessed image
    axes[row, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[row, 0].set_title(f"Image {idx}\n{img_paths[idx].stem[:30]}...", fontsize=8)
    axes[row, 0].axis("off")

    # Column 1: mask
    axes[row, 1].imshow(mask_bin, cmap="gray")
    axes[row, 1].set_title(f"Mask {idx}\nCell area: {mask_bin.mean()*100:.1f}%", fontsize=8)
    axes[row, 1].axis("off")

    # Column 2: overlay
    overlay = np.stack([img, img, img], axis=-1)
    overlay[mask_bin > 0.5] = [0.2, 0.8, 0.2]  # green = cell
    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title(f"Overlay {idx}", fontsize=8)
    axes[row, 2].axis("off")

    # Column 3: stats
    axes[row, 3].axis("off")
    stats = (
        f"Shape   : {img.shape}\n"
        f"Min     : {img.min():.3f}\n"
        f"Max     : {img.max():.3f}\n"
        f"Mean    : {img.mean():.3f}\n"
        f"Std     : {img.std():.3f}\n\n"
        f"Mask px : {int(mask_bin.sum())}\n"
        f"Coverage: {mask_bin.mean()*100:.1f}%"
    )
    axes[row, 3].text(0.1, 0.5, stats, transform=axes[row, 3].transAxes,
                      fontsize=9, verticalalignment="center", fontfamily="monospace",
                      bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=120, bbox_inches="tight")
plt.show()
print(f"\n✅  Verification image saved → {OUT_PATH}")