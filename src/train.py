"""
train.py
--------
Step 3: U-Net Segmentation Model Training

Purpose:
  Train a 2D U-Net to segment cells from brightfield microscopy images.
  The model learns to produce a binary mask (cell vs background) from
  a normalised grayscale input image.

  Class imbalance handled via weighted DiceBCE loss —
  BBBC006 images have only 10-16% cell coverage (mostly background).
  Without weighting, the model would predict all-black and still get low loss.

Input:
  D:/BRIGHT FIELD/data/processed/  ->  768 .npy float32 images (256x256)
  D:/BRIGHT FIELD/data/masks/      ->  768 .png binary masks   (256x256)

Output:
  D:/BRIGHT FIELD/models/unet_best.pth          ->  best checkpoint
  D:/BRIGHT FIELD/models/training_history.json  ->  loss per epoch
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import json
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(r"D:\BRIGHT FIELD")
PROC_DIR  = BASE_DIR / "data" / "processed"
MASK_DIR  = BASE_DIR / "data" / "masks"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
NUM_EPOCHS = 10
BATCH_SIZE = 4


# ── Dataset ───────────────────────────────────────────────────────────────────
class CellDataset(Dataset):
    """
    Loads preprocessed .npy images and binary .png masks.
    Augmentation: horizontal + vertical flip on training set only.
    """
    def __init__(self, img_paths, mask_paths, augment=False):
        self.imgs    = img_paths
        self.masks   = mask_paths
        self.augment = augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img  = np.load(self.imgs[idx]).astype(np.float32)
        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)
        mask = (mask / 255.0).astype(np.float32)

        if self.augment:
            if np.random.rand() > 0.5:
                img  = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if np.random.rand() > 0.5:
                img  = np.flipud(img).copy()
                mask = np.flipud(mask).copy()

        img  = torch.tensor(img[None])
        mask = torch.tensor(mask[None])
        return img, mask


# ── U-Net building blocks ─────────────────────────────────────────────────────
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
    """
    2D U-Net for binary cell segmentation.
    Input : (B, 1, 256, 256) normalised brightfield image
    Output: (B, 1, 256, 256) sigmoid probability map
    """
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


# ── Weighted DiceBCE Loss ─────────────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    """
    Combined Dice + weighted BCE loss.
    pos_weight=6.0 handles class imbalance (10-16% cell coverage).
    """
    def __init__(self, smooth=1.0, pos_weight=6.0):
        super().__init__()
        self.smooth     = smooth
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        weight = torch.ones_like(target)
        weight[target > 0.5] = self.pos_weight
        bce = F.binary_cross_entropy(pred, target, weight=weight)
        p, t = pred.view(-1), target.view(-1)
        dice = 1 - (2 * (p * t).sum() + self.smooth) / \
                   (p.sum() + t.sum() + self.smooth)
        return bce + dice


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device  : {device}")

    img_paths  = sorted(PROC_DIR.glob("*.npy"))
    mask_paths = sorted(MASK_DIR.glob("*.png"))
    print(f"Images  : {len(img_paths)}  |  Masks : {len(mask_paths)}")

    dataset = CellDataset(img_paths, mask_paths, augment=True)
    n_val   = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    print(f"Train   : {n_train}  |  Val : {n_val}")

    model     = UNet(base=32).to(device)
    criterion = DiceBCELoss(pos_weight=6.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=NUM_EPOCHS)

    best_val  = float("inf")
    history   = {"train": [], "val": []}
    save_path = MODEL_DIR / "unet_best.pth"

    print(f"\nTraining for {NUM_EPOCHS} epochs...\n")

    epoch_bar = tqdm(range(1, NUM_EPOCHS + 1), desc="Overall", position=0)

    for epoch in epoch_bar:

        # ── Train ──
        model.train()
        train_loss = 0.0
        train_bar  = tqdm(train_loader,
                          desc=f"Epoch {epoch:02d}/{NUM_EPOCHS} [train]",
                          leave=False, position=1)
        for imgs, masks in train_bar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_bar  = tqdm(val_loader,
                        desc=f"Epoch {epoch:02d}/{NUM_EPOCHS} [val]  ",
                        leave=False, position=1)
        with torch.no_grad():
            for imgs, masks in val_bar:
                imgs, masks = imgs.to(device), masks.to(device)
                loss = criterion(model(imgs), masks)
                val_loss += loss.item()
                val_bar.set_postfix(loss=f"{loss.item():.4f}")
        val_loss /= len(val_loader)
        scheduler.step()

        history["train"].append(round(train_loss, 4))
        history["val"].append(round(val_loss, 4))

        saved = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            saved = "  ✅ saved"

        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_loss:.4f}"
        )
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
              f"train={train_loss:.4f}  "
              f"val={val_loss:.4f}{saved}")

    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✅  Training complete.")
    print(f"    Best val loss : {best_val:.4f}")
    print(f"    Model saved   : {save_path}")


if __name__ == "__main__":
    train()