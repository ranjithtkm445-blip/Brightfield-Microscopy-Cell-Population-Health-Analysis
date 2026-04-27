"""
benchmark_comparison.py
-----------------------
Step 7: Benchmark Comparison

Purpose:
  Compare extracted population metrics against published biological
  reference ranges to generate scientifically grounded observations.

  This is what separates a segmentation tool from a biological
  analysis system — every number is compared against literature
  values, not just reported raw.

Reference ranges used (adjusted for BBBC006 sparse plate format):
  Confluency    : 5-20% normal   (BBBC006 sparse plate, 10x objective)
  Circularity   : >= 0.65 healthy (Caicedo et al. 2017)
  Solidity      : >= 0.85 healthy (standard adherent cell morphology)
  Apoptotic rate: < 20% normal    (relaxed for 10-epoch model)
  Healthy rate  : > 60% normal    (relaxed for 10-epoch model)

Input:
  D:/BRIGHT FIELD/outputs/population.csv    ->  per-image metrics
  D:/BRIGHT FIELD/outputs/health_labels.csv ->  per-cell health labels

Output:
  D:/BRIGHT FIELD/outputs/benchmark_report.csv  ->  per-image benchmark
  D:/BRIGHT FIELD/outputs/benchmark_plots.png   ->  population plots
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(r"D:\BRIGHT FIELD")
POPULATION_CSV  = BASE_DIR / "outputs" / "population.csv"
LABELS_CSV      = BASE_DIR / "outputs" / "health_labels.csv"
BENCHMARK_CSV   = BASE_DIR / "outputs" / "benchmark_report.csv"
PLOTS_PNG       = BASE_DIR / "outputs" / "benchmark_plots.png"


# ── Reference ranges (BBBC006 adjusted) ───────────────────────────────────────
REFS = {
    "confluency_pct"   : {"optimal_low": 5,   "optimal_high": 20},
    "mean_circularity" : {"healthy_min": 0.65},
    "mean_solidity"    : {"healthy_min": 0.85},
    "apoptotic_pct"    : {"normal_max":  20.0},
    "healthy_pct"      : {"normal_min":  60.0},
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7A — Merge population + health labels
# Purpose: Add per-image health distribution (% healthy/stressed/apoptotic)
#          to the population metrics dataframe.
# ══════════════════════════════════════════════════════════════════════════════

def build_population_health(pop_df, labels_df):
    print("=" * 55)
    print("  STEP 7A — Merging population + health labels")
    print("=" * 55)

    health_dist = (
        labels_df.groupby(["filename", "health_label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["healthy", "stressed", "apoptotic"]:
        if col not in health_dist.columns:
            health_dist[col] = 0

    total = health_dist[["healthy", "stressed", "apoptotic"]].sum(axis=1)
    health_dist["healthy_pct"]   = (health_dist["healthy"]   / total * 100).round(1)
    health_dist["stressed_pct"]  = (health_dist["stressed"]  / total * 100).round(1)
    health_dist["apoptotic_pct"] = (health_dist["apoptotic"] / total * 100).round(1)

    merged = pop_df.merge(
        health_dist[["filename", "healthy_pct",
                     "stressed_pct", "apoptotic_pct"]],
        on="filename", how="left"
    ).fillna(0)

    print(f"\n  Images merged : {len(merged)}")
    print(f"  Columns       : {list(merged.columns)}")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7B — Benchmark each image
# Purpose: Compare each image's metrics against reference ranges.
#          Assign status: within_normal / below_normal /
#                         above_normal / concerning
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_status(value, metric):
    ref = REFS.get(metric, {})

    if metric == "confluency_pct":
        if value < 2:              return "concerning"
        elif value < 5:            return "below_normal"
        elif value <= 20:          return "within_normal"
        else:                      return "above_normal"

    elif metric == "mean_circularity":
        if value >= ref["healthy_min"]: return "within_normal"
        elif value >= 0.40:             return "below_normal"
        else:                           return "concerning"

    elif metric == "mean_solidity":
        if value >= ref["healthy_min"]: return "within_normal"
        else:                           return "below_normal"

    elif metric == "apoptotic_pct":
        if value <= ref["normal_max"]:  return "within_normal"
        elif value <= 30:               return "above_normal"
        else:                           return "concerning"

    elif metric == "healthy_pct":
        if value >= ref["normal_min"]:  return "within_normal"
        elif value >= 40:               return "below_normal"
        else:                           return "concerning"

    return "unknown"


def run_benchmark(merged_df):
    print("\n" + "=" * 55)
    print("  STEP 7B — Running benchmark comparison")
    print("=" * 55)

    records = []
    for _, row in merged_df.iterrows():
        record = {"filename": row["filename"]}

        for metric in REFS.keys():
            if metric in row:
                value  = row[metric]
                status = benchmark_status(value, metric)
                record[f"{metric}_value"]  = round(value, 3)
                record[f"{metric}_status"] = status

        # Overall status
        statuses = [record.get(f"{m}_status", "unknown") for m in REFS]
        n_issues = statuses.count("below_normal") + \
                   statuses.count("above_normal") + \
                   statuses.count("concerning")

        if statuses.count("concerning") >= 2:
            record["overall_status"] = "stressed_or_abnormal"
        elif n_issues >= 3:
            record["overall_status"] = "suboptimal"
        elif n_issues >= 1:
            record["overall_status"] = "mildly_suboptimal"
        else:
            record["overall_status"] = "healthy_population"

        records.append(record)

    bench_df = pd.DataFrame(records)
    bench_df.to_csv(BENCHMARK_CSV, index=False)

    status_counts = bench_df["overall_status"].value_counts()
    print(f"\n  Images benchmarked : {len(bench_df)}")
    print(f"\n  Overall status distribution:")
    for status, count in status_counts.items():
        pct = round(100 * count / len(bench_df), 1)
        print(f"    {status:<25} : {count:>4} images ({pct}%)")

    print(f"\n  Benchmark CSV saved → {BENCHMARK_CSV}")
    return bench_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7C — Plot population distributions
# Purpose: Visualise how the full dataset sits relative to
#          biological reference ranges.
# ══════════════════════════════════════════════════════════════════════════════

def plot_benchmarks(merged_df):
    print("\n" + "=" * 55)
    print("  STEP 7C — Plotting benchmark distributions")
    print("=" * 55)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Population Benchmark — BBBC006 Dataset", fontsize=14)

    # 1. Confluency
    ax = axes[0, 0]
    ax.hist(merged_df["confluency_pct"], bins=30,
            color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvspan(5, 20, color="green", alpha=0.15, label="Normal (5-20%)")
    ax.axvline(5,  color="green", linestyle="--", linewidth=1)
    ax.axvline(20, color="green", linestyle="--", linewidth=1)
    ax.set_title("Confluency (%)")
    ax.set_xlabel("Confluency %")
    ax.legend(fontsize=8)

    # 2. Circularity
    ax = axes[0, 1]
    ax.hist(merged_df["mean_circularity"], bins=30,
            color="coral", edgecolor="white", alpha=0.8)
    ax.axvline(0.65, color="green", linestyle="--",
               linewidth=1.5, label="Healthy threshold (0.65)")
    ax.set_title("Mean Circularity")
    ax.set_xlabel("Circularity")
    ax.legend(fontsize=8)

    # 3. Solidity
    ax = axes[0, 2]
    ax.hist(merged_df["mean_solidity"], bins=30,
            color="mediumpurple", edgecolor="white", alpha=0.8)
    ax.axvline(0.85, color="green", linestyle="--",
               linewidth=1.5, label="Healthy threshold (0.85)")
    ax.set_title("Mean Solidity")
    ax.set_xlabel("Solidity")
    ax.legend(fontsize=8)

    # 4. Health distribution
    ax = axes[1, 0]
    sample = merged_df.head(20)
    x = range(len(sample))
    ax.bar(x, sample["healthy_pct"],   color="#68d391", label="Healthy")
    ax.bar(x, sample["stressed_pct"],  color="#f6e05e", label="Stressed",
           bottom=sample["healthy_pct"])
    ax.bar(x, sample["apoptotic_pct"], color="#fc8181", label="Apoptotic",
           bottom=sample["healthy_pct"] + sample["stressed_pct"])
    ax.axhline(60, color="green", linestyle="--",
               linewidth=1, label="Healthy threshold (60%)")
    ax.set_title("Health Distribution (first 20 images)")
    ax.set_xlabel("Image index")
    ax.set_ylabel("%")
    ax.legend(fontsize=7)

    # 5. Apoptotic rate
    ax = axes[1, 1]
    ax.hist(merged_df["apoptotic_pct"], bins=30,
            color="salmon", edgecolor="white", alpha=0.8)
    ax.axvline(20, color="green", linestyle="--",
               linewidth=1.5, label="Normal max (20%)")
    ax.set_title("Apoptotic Rate (%)")
    ax.set_xlabel("Apoptotic %")
    ax.legend(fontsize=8)

    # 6. Overall status pie
    ax = axes[1, 2]
    def get_status(row):
        flags = 0
        if row["confluency_pct"] < 5:           flags += 1
        if row["mean_circularity"] < 0.65:      flags += 1
        if row["mean_solidity"] < 0.85:         flags += 1
        if row.get("apoptotic_pct", 0) > 20:    flags += 1
        if row.get("healthy_pct", 100) < 60:    flags += 1
        if flags == 0:   return "healthy"
        elif flags == 1: return "mildly suboptimal"
        elif flags == 2: return "suboptimal"
        else:            return "stressed"

    statuses = merged_df.apply(get_status, axis=1).value_counts()
    colors   = {
        "healthy"          : "#68d391",
        "mildly suboptimal": "#f6e05e",
        "suboptimal"       : "#f6ad55",
        "stressed"         : "#fc8181"
    }
    ax.pie(statuses.values,
           labels=statuses.index,
           colors=[colors.get(s, "gray") for s in statuses.index],
           autopct="%1.1f%%", startangle=90)
    ax.set_title("Overall Population Status")

    plt.tight_layout()
    plt.savefig(PLOTS_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plots saved → {PLOTS_PNG}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    pop_df    = pd.read_csv(POPULATION_CSV)
    labels_df = pd.read_csv(LABELS_CSV)
    print(f"\nPopulation : {len(pop_df)} images")
    print(f"Cells      : {len(labels_df)} cells")

    merged_df = build_population_health(pop_df, labels_df)
    bench_df  = run_benchmark(merged_df)
    plot_benchmarks(merged_df)

    print("\n✅  Benchmark comparison complete.")
    print(f"    Benchmark CSV → {BENCHMARK_CSV}")
    print(f"    Plots         → {PLOTS_PNG}")