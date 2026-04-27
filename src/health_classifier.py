"""
health_classifier.py
--------------------
Step 6: Health Classification — Auto-label + Train ML Classifier

Purpose:
  Step 6a — Auto-label each cell as Healthy / Stressed / Apoptotic
             using biological reference thresholds from literature.
             (Caicedo et al. 2017, Freshney 2016)

  Step 6b — Train a Random Forest classifier on the 7 morphological
             features to learn the boundary between health states.

  Step 6c — Evaluate classifier performance and save confusion matrix.

  Step 6d — Save trained classifier for use in the Streamlit app.

Why Random Forest:
  - Works well on small tabular datasets (34k cells, 7 features)
  - Interpretable — feature importance shows which features matter most
  - No GPU needed — fast on CPU
  - Handles class imbalance well with class_weight='balanced'

Input:
  D:/BRIGHT FIELD/outputs/features.csv  ->  34,456 cells x 7 features

Output:
  D:/BRIGHT FIELD/models/health_classifier.pkl  ->  trained classifier
  D:/BRIGHT FIELD/outputs/health_labels.csv     ->  features + labels
  D:/BRIGHT FIELD/outputs/classifier_report.png ->  confusion matrix
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(r"D:\BRIGHT FIELD")
FEATURES_CSV   = BASE_DIR / "outputs" / "features.csv"
LABELS_CSV     = BASE_DIR / "outputs" / "health_labels.csv"
REPORT_PNG     = BASE_DIR / "outputs" / "classifier_report.png"
MODEL_PATH     = BASE_DIR / "models" / "health_classifier.pkl"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6A — Auto-labelling
# Purpose: Assign health label to each cell using biological thresholds.
#          These thresholds come from published reference ranges:
#            Caicedo et al. 2017 — BBBC morphology benchmarks
#            Freshney 2016       — Culture of Animal Cells
#
# Rules:
#   Healthy   : circularity >= 0.65 AND solidity >= 0.85 AND
#               mean_intensity >= 0.25 AND area >= 100
#   Apoptotic : circularity <  0.40 OR  solidity <  0.70 OR
#               area < 100 OR mean_intensity < 0.20
#   Stressed  : everything in between
# ══════════════════════════════════════════════════════════════════════════════

def auto_label(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 55)
    print("  STEP 6A — Auto-labelling cells")
    print("=" * 55)

    labels = []
    for _, row in df.iterrows():
        circ  = row["circularity"]
        sol   = row["solidity"]
        inten = row["mean_intensity"]
        area  = row["area"]

        if (circ >= 0.65 and sol >= 0.85 and
                inten >= 0.25 and area >= 100):
            labels.append("healthy")
        elif (circ < 0.40 or sol < 0.70 or
              area < 100 or inten < 0.20):
            labels.append("apoptotic")
        else:
            labels.append("stressed")

    df = df.copy()
    df["health_label"] = labels

    # Distribution
    counts = df["health_label"].value_counts()
    total  = len(df)
    print(f"\n  Total cells : {total}")
    for label, count in counts.items():
        pct = round(100 * count / total, 1)
        print(f"  {label:<12} : {count:>6} cells  ({pct}%)")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6B — Train Random Forest Classifier
# Purpose: Learn the decision boundary between health states
#          from the 7 morphological features.
#          class_weight='balanced' handles label imbalance.
#          80/20 stratified train/test split preserves class ratios.
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "area", "perimeter", "circularity",
    "eccentricity", "solidity",
    "mean_intensity", "std_intensity"
]

def train_classifier(df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("  STEP 6B — Training Random Forest classifier")
    print("=" * 55)

    X  = df[FEATURE_COLS].values
    le = LabelEncoder()
    y  = le.fit_transform(df["health_label"].values)

    print(f"\n  Classes : {list(le.classes_)}")
    print(f"  Features: {FEATURE_COLS}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        stratify=y, random_state=42
    )
    print(f"\n  Train   : {len(X_train)} cells")
    print(f"  Test    : {len(X_test)} cells")

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    print("\n  Training complete.")

    return clf, le, X_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6C — Evaluate Classifier
# Purpose: Print per-class accuracy, F1, and plot confusion matrix.
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_classifier(clf, le, X_test, y_test):
    print("\n" + "=" * 55)
    print("  STEP 6C — Evaluating classifier")
    print("=" * 55)

    y_pred   = clf.predict(X_test)
    acc      = accuracy_score(y_test, y_pred)
    classes  = le.classes_

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"\n  Per-class report:")
    print(classification_report(y_test, y_pred,
                                target_names=classes))

    # Feature importance
    importances = clf.feature_importances_
    print("  Feature importances:")
    for feat, imp in sorted(zip(FEATURE_COLS, importances),
                            key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"    {feat:<18} {imp:.4f}  {bar}")

    # Confusion matrix plot
    cm  = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: confusion matrix
    im = axes[0].imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=axes[0])
    axes[0].set_xticks(range(len(classes)))
    axes[0].set_yticks(range(len(classes)))
    axes[0].set_xticklabels(classes, rotation=15)
    axes[0].set_yticklabels(classes)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, str(cm[i, j]),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
    axes[0].set_title("Confusion matrix — test set")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Right: feature importance
    sorted_idx  = np.argsort(importances)
    axes[1].barh([FEATURE_COLS[i] for i in sorted_idx],
                 importances[sorted_idx], color="steelblue")
    axes[1].set_title("Feature importance")
    axes[1].set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig(REPORT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Report saved → {REPORT_PNG}")

    return acc


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6D — Save Classifier
# Purpose: Save trained classifier + label encoder to disk.
#          Loaded by inference.py and app.py for single-image prediction.
# ══════════════════════════════════════════════════════════════════════════════

def save_classifier(clf, le):
    print("\n" + "=" * 55)
    print("  STEP 6D — Saving classifier")
    print("=" * 55)

    joblib.dump({"clf": clf, "le": le, "features": FEATURE_COLS},
                MODEL_PATH)
    print(f"\n  Saved → {MODEL_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Load features
    print(f"\nLoading features from {FEATURES_CSV}")
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} cells")

    # Step 6a — auto-label
    df = auto_label(df)
    df.to_csv(LABELS_CSV, index=False)
    print(f"\n  Labels saved → {LABELS_CSV}")

    # Step 6b — train
    clf, le, X_test, y_test = train_classifier(df)

    # Step 6c — evaluate
    evaluate_classifier(clf, le, X_test, y_test)

    # Step 6d — save
    save_classifier(clf, le)

    print("\n✅  Health classifier complete.")
    print(f"    Model → {MODEL_PATH}")
    print(f"    Labels→ {LABELS_CSV}")
    print(f"    Report→ {REPORT_PNG}")