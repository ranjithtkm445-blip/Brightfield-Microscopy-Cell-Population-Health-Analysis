"""
inference.py
------------
Step 8: Report Assembly

Purpose:
  End-to-end inference pipeline for a single uploaded image.
  Runs all steps in sequence and assembles a complete biological
  observation report.

  Added features:
    - Confidence scores per cell (Random Forest probability)
    - GradCAM heatmap on U-Net encoder (visual explanation)
    - PDF report generation (downloadable summary)

Pipeline:
  1. Load + preprocess uploaded image
  2. U-Net predicts binary mask + GradCAM heatmap
  3. Connected components → cell instances
  4. skimage.regionprops → 7 morphological features per cell
  5. Random Forest → health label + confidence score per cell
  6. Benchmark comparison → population status
  7. Assemble structured report + PDF

Input:
  Any brightfield image (.tif / .png / .jpg / .npy)
  models/unet_best.pth         ->  trained U-Net
  models/health_classifier.pkl ->  trained RF classifier

Output:
  InferenceReport dataclass
  outputs/report.pdf           ->  PDF report
  outputs/gradcam.png          ->  GradCAM heatmap
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import joblib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage import measure
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm as rcm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage)
from reportlab.lib.enums import TA_CENTER
import os

# ── Paths (HF compatible) ─────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
UNET_PATH       = BASE_DIR / "models" / "unet_best.pth"
CLASSIFIER_PATH = BASE_DIR / "models" / "health_classifier.pkl"
OUTPUT_DIR      = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Report dataclass ──────────────────────────────────────────────────────────
@dataclass
class InferenceReport:
    filename          : str
    n_cells           : int
    confluency_pct    : float
    mean_area         : float
    mean_circularity  : float
    mean_solidity     : float
    mean_intensity    : float
    healthy_pct       : float
    stressed_pct      : float
    apoptotic_pct     : float
    mean_confidence   : float
    overall_status    : str
    observations      : List[str] = field(default_factory=list)
    recommendations   : List[str] = field(default_factory=list)
    caveats           : List[str] = field(default_factory=list)
    overlay_image     : Optional[np.ndarray] = field(default=None)
    gradcam_image     : Optional[np.ndarray] = field(default=None)
    cell_details      : List[Dict] = field(default_factory=list)

    def to_dict(self):
        return {
            "filename"        : self.filename,
            "n_cells"         : self.n_cells,
            "confluency_pct"  : self.confluency_pct,
            "mean_area"       : self.mean_area,
            "mean_circularity": self.mean_circularity,
            "mean_solidity"   : self.mean_solidity,
            "mean_intensity"  : self.mean_intensity,
            "healthy_pct"     : self.healthy_pct,
            "stressed_pct"    : self.stressed_pct,
            "apoptotic_pct"   : self.apoptotic_pct,
            "mean_confidence" : self.mean_confidence,
            "overall_status"  : self.overall_status,
            "observations"    : self.observations,
            "recommendations" : self.recommendations,
            "caveats"         : self.caveats,
        }


# ── U-Net ─────────────────────────────────────────────────────────────────────
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
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear",
                                align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2,
                         diffY//2, diffY-diffY//2])
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


# ── Model loaders ─────────────────────────────────────────────────────────────
_unet       = None
_classifier = None

def load_unet(device):
    global _unet
    if _unet is None:
        model = UNet(base=32).to(device)
        model.load_state_dict(torch.load(UNET_PATH,
                                         map_location=device,
                                         weights_only=True))
        model.eval()
        _unet = model
    return _unet

def load_classifier():
    global _classifier
    if _classifier is None:
        _classifier = joblib.load(CLASSIFIER_PATH)
    return _classifier


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image_input, size=256):
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if p.suffix == ".npy":
            img      = np.load(p).astype(np.float32)
            filename = p.name
            if img.max() <= 1.0:
                return img, filename
        else:
            img = cv2.imread(str(p),
                             cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Cannot read: {p}")
            filename = p.name
    else:
        img      = image_input
        filename = "uploaded_image"

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img   = cv2.resize(img, (size, size),
                       interpolation=cv2.INTER_LANCZOS4)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img   = clahe.apply(img)
    img   = cv2.GaussianBlur(img, (3, 3), 0)
    arr   = img.astype(np.float32) / 255.0
    return arr, filename


# ── GradCAM ───────────────────────────────────────────────────────────────────
def compute_gradcam(img, model, device):
    model.eval()
    tensor      = torch.tensor(img[None, None],
                               requires_grad=False).to(device)
    activations = {}
    gradients   = {}

    def fwd_hook(module, input, output):
        activations["down4"] = output.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients["down4"] = grad_out[0].detach()

    fwd_handle = model.down4.register_forward_hook(fwd_hook)
    bwd_handle = model.down4.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    output = model(tensor)
    loss   = output.mean()
    loss.backward()

    fwd_handle.remove()
    bwd_handle.remove()

    acts    = activations["down4"].squeeze()
    grads   = gradients["down4"].squeeze()
    weights = grads.mean(dim=(1, 2))
    cam     = (weights[:, None, None] * acts).sum(dim=0)
    cam     = F.relu(cam)

    cam = cam.cpu().numpy()
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))

    heatmap = (cm.jet(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    base    = (np.stack([img] * 3, axis=-1) * 255).astype(np.uint8)
    overlay = cv2.addWeighted(base, 0.5, heatmap, 0.5, 0)
    return overlay


# ── Segmentation ──────────────────────────────────────────────────────────────
def segment(img, model, device, threshold=0.5):
    tensor = torch.tensor(img[None, None]).to(device)
    with torch.no_grad():
        prob = model(tensor).squeeze().cpu().numpy()
    return (prob > threshold).astype(np.uint8)


# ── Instance detection ────────────────────────────────────────────────────────
def get_instances(mask, min_area=50, max_area=5000):
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean     = cv2.morphologyEx(
        (mask * 255).astype(np.uint8), cv2.MORPH_OPEN, kernel)
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
    props = measure.regionprops(label_map, intensity_image=img)
    cells = []
    for p in props:
        area        = float(p.area)
        perimeter   = float(p.perimeter) if p.perimeter > 0 else 1.0
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
        region_px   = img[label_map == p.label]
        cells.append({
            "cell_id"       : int(p.label),
            "area"          : area,
            "perimeter"     : perimeter,
            "circularity"   : circularity,
            "eccentricity"  : float(p.eccentricity),
            "solidity"      : float(p.solidity),
            "mean_intensity": float(region_px.mean()),
            "std_intensity" : float(region_px.std()),
        })
    return cells


# ── Health classification + confidence ────────────────────────────────────────
FEATURE_COLS = ["area", "perimeter", "circularity",
                "eccentricity", "solidity",
                "mean_intensity", "std_intensity"]

def classify_health(cells, clf_bundle):
    if not cells:
        return [], {}, []

    clf   = clf_bundle["clf"]
    le    = clf_bundle["le"]
    X     = np.array([[c[f] for f in FEATURE_COLS] for c in cells])
    preds = le.inverse_transform(clf.predict(X))
    proba = clf.predict_proba(X)
    confidences = proba.max(axis=1).tolist()

    health_summary = {"healthy": 0, "stressed": 0, "apoptotic": 0}
    for label in preds:
        health_summary[label] = health_summary.get(label, 0) + 1

    return list(preds), health_summary, confidences


# ── Overlay ───────────────────────────────────────────────────────────────────
def build_overlay(img, label_map, health_preds):
    COLOR = {
        "healthy"  : (0,   200, 0),
        "stressed" : (0,   200, 200),
        "apoptotic": (0,   0,   220),
    }
    rgb = (np.stack([img] * 3, axis=-1) * 255).astype(np.uint8)
    for cell_id, label in enumerate(health_preds, start=1):
        color = COLOR.get(label, (128, 128, 128))
        rgb[label_map == cell_id] = color
    return rgb


# ── Benchmark ─────────────────────────────────────────────────────────────────
def benchmark_status(value, metric):
    if metric == "confluency_pct":
        if value < 2:       return "concerning"
        elif value < 5:     return "below_normal"
        elif value <= 20:   return "within_normal"
        else:               return "above_normal"
    elif metric == "mean_circularity":
        if value >= 0.65:   return "within_normal"
        elif value >= 0.40: return "below_normal"
        else:               return "concerning"
    elif metric == "mean_solidity":
        if value >= 0.85:   return "within_normal"
        else:               return "below_normal"
    elif metric == "apoptotic_pct":
        if value <= 20:     return "within_normal"
        elif value <= 30:   return "above_normal"
        else:               return "concerning"
    elif metric == "healthy_pct":
        if value >= 60:     return "within_normal"
        elif value >= 40:   return "below_normal"
        else:               return "concerning"
    return "unknown"


# ── PDF generation ────────────────────────────────────────────────────────────
def generate_pdf(report, overlay_path, gradcam_path, out_path):
    doc    = SimpleDocTemplate(out_path, pagesize=A4,
                               rightMargin=2*rcm, leftMargin=2*rcm,
                               topMargin=2*rcm, bottomMargin=2*rcm)
    styles = getSampleStyleSheet()
    story  = []

    title_style = ParagraphStyle("title", fontSize=16,
                                 fontName="Helvetica-Bold",
                                 alignment=TA_CENTER, spaceAfter=12)
    story.append(Paragraph("Brightfield Cell Population Analysis Report",
                            title_style))
    story.append(Paragraph(f"File: {report.filename}", styles["Normal"]))
    story.append(Spacer(1, 0.4*rcm))

    status_color = {
        "healthy_population"  : colors.green,
        "mildly_suboptimal"   : colors.orange,
        "suboptimal"          : colors.darkorange,
        "stressed_or_abnormal": colors.red,
    }.get(report.overall_status, colors.grey)

    status_style = ParagraphStyle("status", fontSize=13,
                                  fontName="Helvetica-Bold",
                                  textColor=status_color, spaceAfter=8)
    story.append(Paragraph(
        f"Overall Status: {report.overall_status.replace('_', ' ').upper()}",
        status_style))
    story.append(Spacer(1, 0.3*rcm))

    story.append(Paragraph("Population Metrics", styles["Heading2"]))
    table_data = [
        ["Metric", "Value", "Reference Range"],
        ["Total cells",       str(report.n_cells),               "50–300 cells/FOV"],
        ["Confluency",        f"{report.confluency_pct:.1f}%",   "5–20% (BBBC006)"],
        ["Mean cell area",    f"{report.mean_area:.0f} px²",     "100–3000 px²"],
        ["Mean circularity",  f"{report.mean_circularity:.3f}",  "≥ 0.65 (healthy)"],
        ["Mean solidity",     f"{report.mean_solidity:.3f}",     "≥ 0.85 (healthy)"],
        ["Mean intensity",    f"{report.mean_intensity:.3f}",    "0.25–0.65 a.u."],
        ["Healthy cells",     f"{report.healthy_pct:.1f}%",      "≥ 60%"],
        ["Stressed cells",    f"{report.stressed_pct:.1f}%",     "< 25%"],
        ["Apoptotic cells",   f"{report.apoptotic_pct:.1f}%",    "< 20%"],
        ["Mean confidence",   f"{report.mean_confidence:.3f}",   "0–1 (higher=better)"],
    ]
    t = Table(table_data, colWidths=[6*rcm, 4*rcm, 6*rcm])
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#2d3748")),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, 0), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f7fafc"), colors.white]),
        ("FONTSIZE",       (0, 1), (-1, -1), 9),
        ("GRID",           (0, 0), (-1, -1), 0.5,
         colors.HexColor("#e2e8f0")),
        ("ALIGN",          (1, 0), (-1, -1), "CENTER"),
        ("PADDING",        (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4*rcm))

    story.append(Paragraph("Segmentation Overlay", styles["Heading2"]))
    if os.path.exists(overlay_path):
        story.append(RLImage(overlay_path, width=8*rcm, height=8*rcm))
    story.append(Spacer(1, 0.3*rcm))

    story.append(Paragraph("GradCAM Heatmap", styles["Heading2"]))
    story.append(Paragraph(
        "Regions highlighted in red/yellow indicate areas the U-Net "
        "focused on when predicting cell locations.", styles["Normal"]))
    if os.path.exists(gradcam_path):
        story.append(RLImage(gradcam_path, width=8*rcm, height=8*rcm))
    story.append(Spacer(1, 0.3*rcm))

    story.append(Paragraph("Biological Observations", styles["Heading2"]))
    for obs in report.observations:
        story.append(Paragraph(f"• {obs}", styles["Normal"]))
    story.append(Spacer(1, 0.3*rcm))

    story.append(Paragraph("Recommendations", styles["Heading2"]))
    for rec in report.recommendations:
        story.append(Paragraph(f"• {rec}", styles["Normal"]))
    story.append(Spacer(1, 0.3*rcm))

    story.append(Paragraph("Caveats", styles["Heading2"]))
    for cav in report.caveats:
        story.append(Paragraph(f"⚑  {cav}", styles["Normal"]))

    doc.build(story)


# ── Report assembly ───────────────────────────────────────────────────────────
def build_report(filename, n_cells, confluency, cells,
                 health_summary, confidences, img,
                 label_map, health_preds):

    total         = max(n_cells, 1)
    healthy_pct   = round(100 * health_summary.get("healthy",   0) / total, 1)
    stressed_pct  = round(100 * health_summary.get("stressed",  0) / total, 1)
    apoptotic_pct = round(100 * health_summary.get("apoptotic", 0) / total, 1)
    mean_conf     = round(float(np.mean(confidences)), 4) if confidences else 0.0

    mean_area  = round(np.mean([c["area"]          for c in cells]), 2) if cells else 0
    mean_circ  = round(np.mean([c["circularity"]   for c in cells]), 4) if cells else 0
    mean_sol   = round(np.mean([c["solidity"]      for c in cells]), 4) if cells else 0
    mean_inten = round(np.mean([c["mean_intensity"] for c in cells]), 4) if cells else 0

    metrics  = {
        "confluency_pct"   : confluency,
        "mean_circularity" : mean_circ,
        "mean_solidity"    : mean_sol,
        "apoptotic_pct"    : apoptotic_pct,
        "healthy_pct"      : healthy_pct,
    }
    statuses  = {m: benchmark_status(v, m) for m, v in metrics.items()}
    n_issues  = sum(1 for s in statuses.values()
                    if s in ["below_normal", "above_normal", "concerning"])
    n_concern = sum(1 for s in statuses.values() if s == "concerning")

    if n_concern >= 2:   overall = "stressed_or_abnormal"
    elif n_issues >= 3:  overall = "suboptimal"
    elif n_issues >= 1:  overall = "mildly_suboptimal"
    else:                overall = "healthy_population"

    obs = []
    if statuses["confluency_pct"] == "within_normal":
        obs.append(f"Confluency ({confluency:.1f}%) is within the normal range "
                   f"for BBBC006 sparse plate format (5–20%).")
    elif statuses["confluency_pct"] == "below_normal":
        obs.append(f"Confluency ({confluency:.1f}%) is below normal (5–20%). "
                   f"Possible low seeding density or impaired attachment.")
    else:
        obs.append(f"Confluency ({confluency:.1f}%) is very low — near background.")

    if statuses["mean_circularity"] == "within_normal":
        obs.append(f"Mean circularity ({mean_circ:.3f}) indicates well-rounded "
                   f"healthy cell morphology (threshold ≥ 0.65).")
    else:
        obs.append(f"Mean circularity ({mean_circ:.3f}) is below healthy threshold "
                   f"(0.65) — elongated or irregular cell shapes detected.")

    if statuses["apoptotic_pct"] == "within_normal":
        obs.append(f"Apoptotic fraction ({apoptotic_pct:.1f}%) is within "
                   f"acceptable range (< 20%).")
    else:
        obs.append(f"Apoptotic fraction ({apoptotic_pct:.1f}%) is elevated. "
                   f"Warrants further investigation.")

    if healthy_pct >= 60:
        obs.append(f"Majority of cells ({healthy_pct:.1f}%) show healthy morphology.")
    else:
        obs.append(f"Only {healthy_pct:.1f}% of cells show healthy morphology "
                   f"— below the expected threshold of 60%.")

    obs.append(f"Mean classifier confidence: {mean_conf:.3f} "
               f"({'high' if mean_conf > 0.85 else 'moderate' if mean_conf > 0.70 else 'low'}).")

    recs = []
    if statuses["confluency_pct"] in ["below_normal", "concerning"]:
        recs.append("Verify seeding density and allow additional time "
                    "for cell attachment before imaging.")
    if statuses["apoptotic_pct"] != "within_normal":
        recs.append("Confirm apoptosis with Annexin V / PI staining.")
        recs.append("Check media freshness, CO₂ stability, "
                    "and incubator temperature.")
    if statuses["mean_circularity"] != "within_normal":
        recs.append("Irregular morphology detected — check for cytoskeletal "
                    "stress (osmolarity, pH, mechanical disruption).")
    if not recs:
        recs.append("Population metrics are within reference ranges. "
                    "Standard monitoring schedule is appropriate.")
    recs.append("Single time-point analysis cannot distinguish growth "
                "inhibition from cytotoxicity. "
                "Parallel control imaging recommended.")

    caveats = [
        "Analysis is based on morphological features only. "
        "Functional assays required for definitive conclusions.",
        "U-Net trained for 10 epochs — segmentation may under-detect "
        "cells. Retrain with 50 epochs for improved accuracy.",
        "Reference ranges adjusted for BBBC006 sparse plate format.",
    ]

    cell_details = []
    for cell, label, conf in zip(cells, health_preds, confidences):
        cell_details.append({
            "cell_id"    : cell["cell_id"],
            "health"     : label,
            "confidence" : round(conf, 4),
            "area"       : round(cell["area"], 1),
            "circularity": round(cell["circularity"], 4),
            "solidity"   : round(cell["solidity"], 4),
        })

    overlay = build_overlay(img, label_map, health_preds)

    return InferenceReport(
        filename=filename,
        n_cells=n_cells,
        confluency_pct=round(confluency, 2),
        mean_area=mean_area,
        mean_circularity=mean_circ,
        mean_solidity=mean_sol,
        mean_intensity=mean_inten,
        healthy_pct=healthy_pct,
        stressed_pct=stressed_pct,
        apoptotic_pct=apoptotic_pct,
        mean_confidence=mean_conf,
        overall_status=overall,
        observations=obs,
        recommendations=recs,
        caveats=caveats,
        overlay_image=overlay,
        cell_details=cell_details,
    )


# ── Full pipeline ─────────────────────────────────────────────────────────────
def run_inference(image_input, size=256,
                  threshold=0.5,
                  save_pdf=True,
                  save_gradcam=True) -> InferenceReport:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img, filename = preprocess(image_input, size=size)
    unet          = load_unet(device)
    mask          = segment(img, unet, device, threshold=threshold)
    confluency    = round(float(mask.mean() * 100), 2)

    gradcam_img  = compute_gradcam(img, unet, device)
    gradcam_path = str(OUTPUT_DIR / "gradcam.png")
    if save_gradcam:
        cv2.imwrite(gradcam_path,
                    cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR))

    label_map = get_instances(mask)
    n_cells   = int(label_map.max())
    cells     = extract_features(img, label_map)

    clf_bundle                            = load_classifier()
    health_preds, health_summary, confidences = classify_health(
        cells, clf_bundle)

    report = build_report(filename, n_cells, confluency,
                          cells, health_summary, confidences,
                          img, label_map, health_preds)
    report.gradcam_image = gradcam_img

    overlay_path = str(OUTPUT_DIR / "overlay.png")
    if report.overlay_image is not None:
        cv2.imwrite(overlay_path,
                    cv2.cvtColor(report.overlay_image, cv2.COLOR_RGB2BGR))

    if save_pdf:
        pdf_path = str(OUTPUT_DIR / "report.pdf")
        generate_pdf(report, overlay_path, gradcam_path, pdf_path)

    return report


# ── Test run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_img = sorted(
        (BASE_DIR / "data" / "processed").glob("*.npy"))[0]
    print(f"Testing on: {test_img.name}\n")
    report = run_inference(test_img)
    print(f"Cells          : {report.n_cells}")
    print(f"Confluency     : {report.confluency_pct}%")
    print(f"Healthy        : {report.healthy_pct}%")
    print(f"Overall status : {report.overall_status}")
    print(f"Confidence     : {report.mean_confidence}")
    print("\n✅  Inference complete.")