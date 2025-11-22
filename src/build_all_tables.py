import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, name: str):
    """Save table as CSV and LaTeX (optional)."""
    csv_path = OUT / f"{name}.csv"
    tex_path = OUT / f"{name}.tex"

    df.to_csv(csv_path, index=False)

    # If you are using LaTeX tables in the paper, uncomment this:
    try:
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(df.to_latex(index=False, escape=True))
    except Exception as e:
        print(f"Could not write LaTeX for {name}: {e}")

    print(f"Saved {name} -> {csv_path}")


def build_table1():
    df = pd.read_csv(RAW / "table1_model_performance.csv")
    # Ensure accuracy is numeric for sorting and sort by dataset then accuracy
    df["Accuracy (%)"] = pd.to_numeric(df["Accuracy (%)"], errors="coerce")
    df = df.sort_values(["Dataset", "Accuracy (%)"], ascending=[True, False])
    save_table(df, "table1_model_performance")


def build_table2():
    df = pd.read_csv(RAW / "table2_hardware_speeds.csv")
    df["Inference Speed (img/s)"] = pd.to_numeric(df["Inference Speed (img/s)"], errors="coerce")
    df = df.sort_values(["Hardware", "Inference Speed (img/s)"], ascending=[True, False])
    save_table(df, "table2_hardware_speeds")


def build_table3():
    df = pd.read_csv(RAW / "table3_datasets.csv")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.sort_values("Year")
    save_table(df, "table3_datasets")


def build_table4():
    df = pd.read_csv(RAW / "table4_challenges.csv")
    save_table(df, "table4_challenges")


def build_table5():
    df = pd.read_csv(RAW / "table5_future_research.csv")
    save_table(df, "table5_future_research")


def build_table6():
    df = pd.read_csv(RAW / "table6_implementation_phases.csv")
    save_table(df, "table6_implementation_phases")


def build_table7():
    df = pd.read_csv(RAW / "table7_evolution_technologies.csv")
    # Sort by extracted numeric year from the era field
    df["era_numeric"] = df["Era"].astype(str).str.extract(r"(\\d{4})").astype(float)
    df = df.sort_values("era_numeric")
    df = df.drop(columns=["era_numeric"])
    save_table(df, "table7_evolution_technologies")


if __name__ == "__main__":
    build_table1()
    build_table2()
    build_table3()
    build_table4()
    build_table5()
    build_table6()
    build_table7()
