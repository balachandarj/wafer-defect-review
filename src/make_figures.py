import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# ---------- Figure 1: Accuracy vs Computational Cost ----------
def make_figure1():
    """
    Figure 1: Accuracy vs Computational Cost Trade-off
    Uses: table1_model_performance.csv
    """
    df = pd.read_csv(RAW / "table1_model_performance.csv")

    # Normalize column names and units from the raw table.
    df = df.rename(columns={
        "Model": "model",
        "Accuracy (%)": "accuracy_pct",
        "FLOPs": "flops_raw",
    })
    df["accuracy_pct"] = pd.to_numeric(df["accuracy_pct"], errors="coerce")

    def to_flops_b(val):
        """Convert strings like '4.1B' or '1.2M' to billions."""
        if pd.isna(val):
            return float("nan")
        s = str(val).strip().upper()
        factor = 1.0
        if s.endswith("B"):
            s = s[:-1]
        elif s.endswith("G"):  # treat G as billions
            s = s[:-1]
        elif s.endswith("M"):
            s = s[:-1]
            factor = 1 / 1000.0
        return pd.to_numeric(s, errors="coerce") * factor

    df["flops_b"] = df["flops_raw"].apply(to_flops_b)

    # flops_b on x-axis, accuracy on y-axis
    plt.figure()
    for _, row in df.iterrows():
        plt.scatter(row["flops_b"], row["accuracy_pct"])
        plt.text(row["flops_b"], row["accuracy_pct"], row["model"],
                 fontsize=8, ha="left", va="bottom")

    plt.xlabel("FLOPs (billions)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Computational Cost for Wafer Defect Detection Models")
    plt.grid(True)
    plt.tight_layout()

    out = OUT / "figure1_accuracy_vs_cost.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


# ---------- Figure 2: Detection Evolution (Year vs Accuracy) ----------
def make_figure2():
    """
    Figure 2: Wafer Defect Detection Evolution (Year vs Accuracy)
    Uses: table8_detection_evolution.csv
    """
    df = pd.read_csv(RAW / "table8_detection_evolution.csv")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["accuracy_pct"] = pd.to_numeric(df["accuracy_pct"], errors="coerce")
    df["defect_size_nm"] = pd.to_numeric(df["defect_size_nm"], errors="coerce")

    df = df.sort_values("year")

    palette = {
        "Manual": "#56b2c9",
        "Optical": "#f3c178",
        "E-beam": "#f38181",
        "Digital": "#7cb7c6",
        "ML": "#f6d55c",
        "DL": "#d64255",
        "Advanced DL": "#c44c9d",
        "Transformers": "#7a4f9c",
        "Modern AI": "#4f7d5f",
    }
    colors = df["category"].map(palette).fillna("#777777")

    plt.figure()
    plt.scatter(df["year"], df["accuracy_pct"], s=140, c=colors, alpha=0.9, edgecolor="none")
    plt.plot(df["year"], df["accuracy_pct"], color="#888888", linestyle="--", linewidth=1)

    for _, row in df.iterrows():
        plt.text(row["year"], row["accuracy_pct"] + 0.5, row["label"],
                 ha="center", va="bottom", fontsize=8)

    plt.xlabel("Year")
    plt.ylabel("Accuracy (%)")
    plt.title("Wafer Defect Detection Evolution")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = OUT / "figure2_timeline_evolution.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


# ---------- Figure 3: Dataset Comparison ----------
def make_figure3():
    """
    Figure 3: Samples vs Classes for Major Datasets
    Uses: table3_datasets.csv
    """
    df = pd.read_csv(RAW / "table3_datasets.csv")

    df = df.rename(columns={
        "Dataset": "dataset",
        "Year": "year",
        "Total Samples": "total_samples",
        "Classes": "classes",
        "Source": "source",
        "Data Type": "data_type",
        "Resolution": "resolution",
        "Label Type": "label_type",
        "Class Balance": "class_balance",
    })

    # Ensure numerics
    df["total_samples"] = pd.to_numeric(df["total_samples"].astype(str).str.replace(",", ""), errors="coerce")
    df["classes"] = pd.to_numeric(df["classes"], errors="coerce")

    plt.figure()
    plt.scatter(df["total_samples"], df["classes"], s=120)

    for _, row in df.iterrows():
        plt.text(row["total_samples"], row["classes"] + 0.05, row["dataset"],
                 ha="center", va="bottom", fontsize=8)

    # Log-scale X with fixed tick labels
    xticks = [1e3, 1e4, 1e5, 1e6]
    ax = plt.gca()
    ax.set_xscale("log")
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",000", "K") if x < 1e6 else "1M"))

    plt.xlabel("Samples (log scale)")
    plt.ylabel("Classes")
    plt.title("Datasets: Samples vs Classes")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = OUT / "figure3_dataset_comparison.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    make_figure1()
    make_figure2()
    make_figure3()
