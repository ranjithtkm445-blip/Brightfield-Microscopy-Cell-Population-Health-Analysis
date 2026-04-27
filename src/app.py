"""
app.py
------
Step 9: Streamlit Web Application

Purpose:
  Interactive web interface for the Brightfield Cell Analysis System.
  User selects a preloaded unseen sample image from the sidebar.
  The app runs the full inference pipeline and displays:
    - Segmentation overlay (green=healthy, yellow=stressed, red=apoptotic)
    - GradCAM heatmap (U-Net attention visualisation)
    - Population metrics with benchmark comparison
    - Health distribution chart
    - Biological observations (prominently displayed)
    - Downloadable PDF and JSON reports

Usage:
  Local  : streamlit run src/app.py
  HF     : streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# ── Paths (HF compatible) ─────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
SAMPLE_DIR = BASE_DIR / "samples"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import run_inference

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brightfield Cell Analyser",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0a0e1a; }

.hero-title {
  font-size: 2.4rem;
  font-weight: 700;
  background: linear-gradient(135deg, #63b3ed, #9f7aea, #68d391);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 0.2rem;
}
.hero-sub {
  font-size: 1rem;
  color: #718096;
  margin-bottom: 2rem;
}

.status-banner {
  padding: 1.2rem 1.8rem;
  border-radius: 14px;
  margin-bottom: 1.5rem;
  border-left: 6px solid;
}
.status-healthy    { background:#0d2e1a; border-color:#68d391; color:#68d391; }
.status-mild       { background:#2d2a0d; border-color:#f6e05e; color:#f6e05e; }
.status-suboptimal { background:#2d1f0d; border-color:#f6ad55; color:#f6ad55; }
.status-stressed   { background:#2d0d0d; border-color:#fc8181; color:#fc8181; }

.status-banner h2  { margin:0; font-size:1.3rem; font-weight:700; }
.status-banner p   { margin:0.3rem 0 0; font-size:0.9rem; opacity:0.8; color:#a0aec0; }

.obs-card {
  background: #111827;
  border: 1px solid #1f2937;
  border-left: 4px solid #63b3ed;
  border-radius: 10px;
  padding: 1rem 1.2rem;
  margin-bottom: 0.7rem;
  font-size: 0.95rem;
  color: #e2e8f0;
  line-height: 1.6;
}
.rec-card {
  background: #111827;
  border: 1px solid #1f2937;
  border-left: 4px solid #68d391;
  border-radius: 10px;
  padding: 1rem 1.2rem;
  margin-bottom: 0.7rem;
  font-size: 0.95rem;
  color: #e2e8f0;
  line-height: 1.6;
}
.cav-card {
  background: #111827;
  border: 1px solid #1f2937;
  border-left: 4px solid #f6ad55;
  border-radius: 10px;
  padding: 0.8rem 1.2rem;
  margin-bottom: 0.5rem;
  font-size: 0.85rem;
  color: #a0aec0;
  line-height: 1.5;
}

.metric-box {
  background: #111827;
  border: 1px solid #1f2937;
  border-radius: 12px;
  padding: 1.1rem 1rem;
  text-align: center;
}
.metric-box .label {
  font-size: 0.75rem;
  color: #718096;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.3rem;
}
.metric-box .value {
  font-size: 1.6rem;
  font-weight: 700;
  color: #e2e8f0;
}
.metric-box .unit {
  font-size: 0.75rem;
  color: #4a5568;
}

.section-header {
  font-size: 1.1rem;
  font-weight: 600;
  color: #63b3ed;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin: 2rem 0 1rem;
  padding-bottom: 0.4rem;
  border-bottom: 1px solid #1f2937;
}

.tag-healthy   { background:#0d2e1a; color:#68d391; padding:2px 10px;
                 border-radius:20px; font-size:0.78rem; font-weight:600; }
.tag-stressed  { background:#2d2a0d; color:#f6e05e; padding:2px 10px;
                 border-radius:20px; font-size:0.78rem; font-weight:600; }
.tag-apoptotic { background:#2d0d0d; color:#fc8181; padding:2px 10px;
                 border-radius:20px; font-size:0.78rem; font-weight:600; }

div[data-baseweb="select"] > div {
    border-color: #68d391 !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: #68d391 !important;
    box-shadow: 0 0 0 2px rgba(104,211,145,0.3) !important;
}
div[data-baseweb="select"] > div:hover {
    border-color: #68d391 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def numpy_to_bytes(arr):
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    _, buf = cv2.imencode(".png", arr)
    return buf.tobytes()


def health_donut(healthy, stressed, apoptotic):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    sizes  = [max(healthy, 0.01), max(stressed, 0.01), max(apoptotic, 0.01)]
    colors = ["#68d391", "#f6e05e", "#fc8181"]
    labels = [f"Healthy\n{healthy:.0f}%",
              f"Stressed\n{stressed:.0f}%",
              f"Apoptotic\n{apoptotic:.0f}%"]
    wedges, _ = ax.pie(
        sizes, colors=colors, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="#0a0e1a", linewidth=2)
    )
    ax.legend(wedges, labels, loc="lower center",
              bbox_to_anchor=(0.5, -0.18), ncol=3,
              fontsize=8, framealpha=0,
              labelcolor="white")
    plt.tight_layout()
    return fig


def metric_box(label, value, unit=""):
    return f"""
    <div class="metric-box">
      <div class="label">{label}</div>
      <div class="value">{value}<span class="unit"> {unit}</span></div>
    </div>
    """


def status_banner(status, filename):
    css_map = {
        "healthy_population"  : "status-healthy",
        "mildly_suboptimal"   : "status-mild",
        "suboptimal"          : "status-suboptimal",
        "stressed_or_abnormal": "status-stressed",
    }
    desc_map = {
        "healthy_population"  : "Cell population metrics are within reference ranges.",
        "mildly_suboptimal"   : "Minor deviations from expected healthy culture parameters.",
        "suboptimal"          : "Several metrics fall outside reference ranges.",
        "stressed_or_abnormal": "Multiple indicators of cellular stress detected.",
    }
    css   = css_map.get(status, "status-mild")
    label = status.replace("_", " ").upper()
    desc  = desc_map.get(status, "")
    return f"""
    <div class="status-banner {css}">
      <h2>{label}</h2>
      <p>{desc} &nbsp;·&nbsp; {filename}</p>
    </div>
    """


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Sample Images")
    st.caption("10 unseen images — not used in training")

    sample_files = sorted(SAMPLE_DIR.glob("*_w1*.tif")) \
        if SAMPLE_DIR.exists() else []
    sample_names = ["— choose an image —"] + \
        [p.name[:45] for p in sample_files]

    selected = st.selectbox("Select image", sample_names,
                            label_visibility="collapsed",
                            key="sample_select")

    st.divider()
    st.markdown("## ⚙️ Settings")
    threshold    = st.slider("Segmentation threshold", 0.3, 0.8, 0.5, 0.05)
    image_size   = st.selectbox("Resolution", [256, 512], index=0)
    show_gradcam = st.checkbox("Show GradCAM", value=True)
    show_cells   = st.checkbox("Show per-cell table", value=False)

    st.divider()
    st.caption(
        "**Dataset:** BBBC006 z_16\n\n"
        "**Model:** U-Net · DiceBCE Loss\n\n"
        "**Classifier:** Random Forest (99.96% acc)\n\n"
        "**References:** Caicedo 2017 · Freshney 2016"
    )


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🔬 Brightfield Cell Analysis</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">U-Net segmentation · Health classification · '
    'Benchmark-referenced biological insight</div>',
    unsafe_allow_html=True
)

# ── Resolve input ─────────────────────────────────────────────────────────────
tmp_path = None
if selected != "— choose an image —":
    idx      = sample_names.index(selected) - 1
    tmp_path = str(sample_files[idx])

# ── Run analysis ──────────────────────────────────────────────────────────────
if tmp_path:
    with st.spinner("Running full pipeline..."):
        try:
            report = run_inference(
                tmp_path,
                size=image_size,
                threshold=threshold,
                save_pdf=True,
                save_gradcam=True,
            )
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    # ── Status banner ─────────────────────────────────────────────────────────
    st.markdown(status_banner(report.overall_status, report.filename),
                unsafe_allow_html=True)

    # ── Images ────────────────────────────────────────────────────────────────
    if show_gradcam:
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.markdown("**Segmentation overlay**")
            st.caption("🟢 Healthy &nbsp; 🟡 Stressed &nbsp; 🔴 Apoptotic")
            if report.overlay_image is not None:
                st.image(numpy_to_bytes(report.overlay_image),
                         use_container_width=True)
        with img_col2:
            st.markdown("**GradCAM — U-Net attention map**")
            st.caption("Red/yellow = regions the model focused on")
            if report.gradcam_image is not None:
                st.image(numpy_to_bytes(report.gradcam_image),
                         use_container_width=True)
    else:
        if report.overlay_image is not None:
            st.image(numpy_to_bytes(report.overlay_image),
                     use_container_width=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Population Metrics</div>',
                unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
    m1.markdown(metric_box("Cells",       report.n_cells,                  ""),    unsafe_allow_html=True)
    m2.markdown(metric_box("Confluency",  f"{report.confluency_pct:.1f}",  "%"),   unsafe_allow_html=True)
    m3.markdown(metric_box("Area",        f"{report.mean_area:.0f}",       "px²"), unsafe_allow_html=True)
    m4.markdown(metric_box("Confidence",  f"{report.mean_confidence:.2f}", ""),    unsafe_allow_html=True)
    m5.markdown(metric_box("Circularity", f"{report.mean_circularity:.3f}",""),    unsafe_allow_html=True)
    m6.markdown(metric_box("Solidity",    f"{report.mean_solidity:.3f}",   ""),    unsafe_allow_html=True)
    m7.markdown(metric_box("Intensity",   f"{report.mean_intensity:.3f}",  ""),    unsafe_allow_html=True)
    m8.markdown(metric_box("Healthy",     f"{report.healthy_pct:.0f}",     "%"),   unsafe_allow_html=True)

    # ── Health distribution ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Health Distribution</div>',
                unsafe_allow_html=True)

    donut_col, tag_col = st.columns([1, 2])
    with donut_col:
        fig = health_donut(report.healthy_pct,
                           report.stressed_pct,
                           report.apoptotic_pct)
        st.pyplot(fig, use_container_width=True)
    with tag_col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            f'<span class="tag-healthy">Healthy &nbsp; {report.healthy_pct:.1f}%</span>'
            f' &nbsp; '
            f'<span class="tag-stressed">Stressed &nbsp; {report.stressed_pct:.1f}%</span>'
            f' &nbsp; '
            f'<span class="tag-apoptotic">Apoptotic &nbsp; {report.apoptotic_pct:.1f}%</span>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"**{report.n_cells}** cells detected across the field of view. "
            f"Classifier mean confidence: **{report.mean_confidence:.3f}** "
            f"({'high' if report.mean_confidence > 0.85 else 'moderate'})."
        )

    # ── Observations ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Biological Observations</div>',
                unsafe_allow_html=True)
    for obs in report.observations:
        st.markdown(f'<div class="obs-card">🔬 &nbsp; {obs}</div>',
                    unsafe_allow_html=True)

    # ── Recommendations ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Recommendations</div>',
                unsafe_allow_html=True)
    for rec in report.recommendations:
        st.markdown(f'<div class="rec-card">✅ &nbsp; {rec}</div>',
                    unsafe_allow_html=True)

    # ── Caveats ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Caveats</div>',
                unsafe_allow_html=True)
    for cav in report.caveats:
        st.markdown(f'<div class="cav-card">⚑ &nbsp; {cav}</div>',
                    unsafe_allow_html=True)

    # ── Per-cell table ────────────────────────────────────────────────────────
    if show_cells and report.cell_details:
        st.markdown('<div class="section-header">Per-cell Details</div>',
                    unsafe_allow_html=True)
        df = pd.DataFrame(report.cell_details)
        st.dataframe(df, use_container_width=True)

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Downloads</div>',
                unsafe_allow_html=True)
    dl1, dl2, _ = st.columns([1, 1, 2])

    pdf_path = OUTPUT_DIR / "report.pdf"
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            dl1.download_button(
                label="⬇ PDF Report",
                data=f.read(),
                file_name=f"cell_analysis_{Path(report.filename).stem}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    dl2.download_button(
        label="⬇ JSON Report",
        data=json.dumps(report.to_dict(), indent=2),
        file_name=f"cell_analysis_{Path(report.filename).stem}.json",
        mime="application/json",
        use_container_width=True,
    )

else:
    # ── Landing page ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0 1rem;">
      <div style="font-size:4rem;">🔬</div>
      <div style="font-size:1.3rem; color:#a0aec0; margin-top:1rem;">
        Select a sample image from the sidebar to begin analysis
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">What this system does</div>',
                unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    f1.markdown("""
    <div class="metric-box">
      <div style="font-size:2rem">🧬</div>
      <div style="color:#63b3ed;font-weight:600;margin:0.5rem 0">Segmentation</div>
      <div style="color:#718096;font-size:0.85rem">U-Net predicts cell vs background mask</div>
    </div>""", unsafe_allow_html=True)
    f2.markdown("""
    <div class="metric-box">
      <div style="font-size:2rem">🏥</div>
      <div style="color:#68d391;font-weight:600;margin:0.5rem 0">Classification</div>
      <div style="color:#718096;font-size:0.85rem">Random Forest classifies cell health state</div>
    </div>""", unsafe_allow_html=True)
    f3.markdown("""
    <div class="metric-box">
      <div style="font-size:2rem">📊</div>
      <div style="color:#9f7aea;font-weight:600;margin:0.5rem 0">Benchmarking</div>
      <div style="color:#718096;font-size:0.85rem">Metrics compared to published reference ranges</div>
    </div>""", unsafe_allow_html=True)
    f4.markdown("""
    <div class="metric-box">
      <div style="font-size:2rem">📄</div>
      <div style="color:#f6ad55;font-weight:600;margin:0.5rem 0">Report</div>
      <div style="color:#718096;font-size:0.85rem">PDF + JSON report with recommendations</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Reference ranges</div>',
                unsafe_allow_html=True)
    refs = pd.DataFrame([
        ["Confluency",     "5–20%",  "BBBC006 sparse plate format"],
        ["Circularity",    "≥ 0.65", "Caicedo et al. 2017"],
        ["Solidity",       "≥ 0.85", "Standard adherent cell morphology"],
        ["Apoptotic rate", "< 20%",  "Normal culture baseline"],
        ["Healthy rate",   "≥ 60%",  "Normal culture baseline"],
    ], columns=["Metric", "Normal Range", "Source"])
    st.table(refs)