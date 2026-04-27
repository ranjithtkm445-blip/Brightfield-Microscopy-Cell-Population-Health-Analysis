# Brightfield-Microscopy-Cell-Population-Health-Analysis
Brightfield Cell Population Health Analysis
U-Net Segmentation · Random Forest Health Classification · Benchmark-Referenced Biological Insight
Dataset: BBBC006 z_16 | PyTorch | Streamlit | MIT License
https://huggingface.co/spaces/Ranjith445/brightfield-cell-analysis

Problem Statement
Brightfield microscopy is the most widely used imaging modality in biology labs — no fluorescence equipment or specialised staining required. However:

Cell boundaries have low contrast, making manual counting error-prone and subjective
Manually analysing hundreds of images per experiment is time-consuming
Extracting quantitative biological insight requires expert domain knowledge

This project builds an automated pipeline from raw grayscale image to a structured biological observation report grounded in published reference ranges.

Pipeline (see table in docx)

Dataset
Source: Broad Bioimage Benchmark Collection (BBBC006)

Images: BBBC006_v1_images_z_16.zip — in-focus z-plane (z=16 of 32), w1 brightfield channel only
Labels: BBBC006_v1_labels.zip — ground-truth binary segmentation masks
Size: 768 image/mask pairs, 256x256 pixels after preprocessing
Cell line: Human U2OS cells under normal culture conditions

Note: BBBC006 contains no drug treatment labels or experimental conditions. All analysis is strictly descriptive and observational — not causal inference.

Technical Details
U-Net Architecture — 4-level encoder-decoder, DiceBCE loss, pos_weight=6.0, AdamW, CosineAnnealingLR, best val loss 0.2322
Health Classifier (Random Forest)

Model: RandomForestClassifier — n_estimators=100, max_depth=10, class_weight='balanced'
Auto-labelling: biological thresholds from Caicedo et al. 2017 and Freshney 2016
Training: 80/20 stratified split on 34,456 cells extracted from 768 images
Accuracy: 99.96% on test set
Top features: circularity (0.287), area (0.259), perimeter (0.200), solidity (0.140)
Saved to: models/health_classifier.pkl via joblib

Biological Benchmark Reference Ranges — Confluency 5–20%, Circularity ≥0.65, Solidity ≥0.85, Apoptotic <20%, Healthy ≥60%

Results

U-Net best val loss: 0.2322 (10 epochs, CPU)
Health classifier accuracy: 99.96%
Total cells extracted: 34,456 across 768 images
Healthy population images: 61.8%
Mildly suboptimal: 33.3%
Stressed or abnormal: 2.6%
Mean confidence score: 0.99 (high)


Limitations

No experimental labels — BBBC006 has no drug treatment or time-course metadata. Analysis is observational only.
Single time-point — cannot distinguish growth inhibition from cytotoxicity without parallel control imaging.
10-epoch training — U-Net under-segments cells at low confluency. Retrain with 50 epochs for production.
Rule-based auto-labels — health classifier trained on threshold-derived labels, not ground-truth biological annotations.
Reference ranges adjusted for BBBC006 sparse plate format — may not generalise to other imaging conditions.

Stating these limitations explicitly is intentional. Understanding the gap between a computational proxy and biological ground truth is a core competency for biomedical AI work.

Related Projects

Brain Tumor MRI Segmentation + Survival Risk — BraTS 2020, U-Net + FusionNet — https://huggingface.co/spaces/Ranjith445/brain-tumor-ai
COVID-19 CT Severity Prediction — ResNet18, GradCAM, severity staging — https://huggingface.co/spaces/Ranjith445
ECG Arrhythmia Detection — 1-D CNN, multi-class classification — https://huggingface.co/spaces/Ranjith445
BBBC021 Fluorescence MoA Prediction — ResNet18 + MLP fusion — https://huggingface.co/spaces/Ranjith445/bbbc021-moa-predictor


Author
M. Ranjith Kumar · Biomedical AI Portfolio · Hugging Face: https://huggingface.co/Ranjith445
GitHub: https://github.com/ranjithtkm445-blip

