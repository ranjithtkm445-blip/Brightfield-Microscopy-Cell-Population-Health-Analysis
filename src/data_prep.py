"""
data_prep.py
------------
BBBC006 brightfield microscopy dataset preparation.

Folder structure on disk:
  D:\BRIGHT FIELD\
    BBBC006_v1_images_z_16\
      BBBC006_v1_images_z_16\
        mcf-z-stacks-03212011_a01_s1_w1<uuid>.tif   <- brightfield (w1)
        mcf-z-stacks-03212011_a01_s1_w2<uuid>.tif   <- fluorescence (w2, ignored)
        ...
    BBBC006_v1_labels\
      BBBC006_v1_labels\
        mcf-z-stacks-03212011_a01_s1.png             <- binary mask
        ...

Pairing logic:
  image stem:  mcf-z-stacks-03212011_a01_s1_w1<uuid>
  label stem:  mcf-z-stacks-03212011_a01_s1
  match key:   everything before _w1  (i.e. strip _w1<uuid> suffix)
"""

from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import re

# ── Raw data locations ────────────────────────────────────────────────────────
IMAGES_DIR = Path(r"D:\BRIGHT FIELD\BBBC006_v1_images_z_16\BBBC006_v1_images_z_16")
LABELS_DIR = Path(r"D:\BRIGHT FIELD\BBBC006_v1_labels\BBBC006_v1_labels")

# ── Output locations ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
MASK_DIR     = PROJECT_ROOT / "data" / "masks"

PROC_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)


# ── Pairing ───────────────────────────────────────────────────────────────────
def get_stem_key(filename: str) -> str:
    return re.sub(r'_w\d.*$', '', filename)


def build_pairs(max_n: int = None):
    all_tifs  = sorted(IMAGES_DIR.glob("*.tif"))
    w1_images = [p for p in all_tifs if "_w1" in p.stem]

    label_lookup = {}
    for lp in LABELS_DIR.glob("*.png"):
        label_lookup[lp.stem] = lp

    pairs = []
    missing = 0
    for img_path in w1_images:
        key = get_stem_key(img_path.stem)
        if key in label_lookup:
            pairs.append((img_path, label_lookup[key]))
        else:
            missing += 1

    if missing:
        print(f"⚠  {missing} images had no matching label — skipped.")

    if max_n:
        pairs = pairs[:max_n]

    print(f"✅  {len(pairs)} matched image/mask pairs found.")
    return pairs


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_image(img_path: Path, out_path: Path, size: int = 512) -> bool:
    img = cv2.imread(str(img_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  ✗  Could not read: {img_path.name}")
        return False

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    img = cv2.GaussianBlur(img, (3, 3), 0)

    arr = img.astype(np.float32) / 255.0
    np.save(out_path, arr)
    return True


def preprocess_mask(mask_path: Path, out_path: Path, size: int = 512) -> bool:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(str(out_path), mask)
    return True


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_preprocessing(max_n: int = None, size: int = 512):
    print("=" * 50)
    print("  BBBC006 Preprocessing Pipeline")
    print("=" * 50)

    pairs = build_pairs(max_n=max_n)
    if not pairs:
        print("✗  No pairs found. Check IMAGES_DIR and LABELS_DIR paths.")
        return [], []

    img_out_paths  = []
    mask_out_paths = []
    skipped = 0

    for img_path, mask_path in tqdm(pairs, desc="Preprocessing"):
        key      = get_stem_key(img_path.stem)
        img_out  = PROC_DIR / f"{key}.npy"
        mask_out = MASK_DIR  / f"{key}_mask.png"

        ok_img  = preprocess_image(img_path,  img_out,  size=size)
        ok_mask = preprocess_mask(mask_path, mask_out, size=size)

        if ok_img and ok_mask:
            img_out_paths.append(img_out)
            mask_out_paths.append(mask_out)
        else:
            skipped += 1

    print(f"\n✅  Done.")
    print(f"    Processed : {len(img_out_paths)} pairs")
    print(f"    Skipped   : {skipped}")
    print(f"    Images → {PROC_DIR}")
    print(f"    Masks  → {MASK_DIR}")
    return img_out_paths, mask_out_paths


if __name__ == "__main__":
    run_preprocessing(max_n=None, size=512)