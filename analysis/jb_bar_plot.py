import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# CONFIG
# ==========================================
INPUT_DIR = "./evaluated_csvs"
OUTPUT_DIR = "./Plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

# ==========================================
# PROCESS ALL FILES
# ==========================================
csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

print(f"--> Found {len(csv_files)} evaluated CSV files")

for csv_file in csv_files:
    input_path = os.path.join(INPUT_DIR, csv_file)
    base_name = csv_file.replace(".csv", "")
    output_plot = os.path.join(
        OUTPUT_DIR,
        f"zero_vs_nonzero_analysis_{base_name}.png"
    )

    print("\n" + "=" * 70)
    print(f" ANALYZING FILE: {csv_file}")
    print("=" * 70)

    df = pd.read_csv(input_path)

    # ---------------------------------------------------------
    # ANALYSIS: Zero vs Non-Zero Alpha
    # ---------------------------------------------------------
    df["Experiment_Group"] = df["Alpha"].apply(
        lambda x: "Baseline (Alpha=0)" if x == 0.0 else "Attack (Alpha>0)"
    )

    comparison = (
        df.groupby("Experiment_Group")["Jailbreak_Success"]
        .mean()
        * 100
    )

    print("\nJAILBREAK SUCCESS ANALYSIS")
    print("-" * 70)
    print(comparison.to_string(float_format="{:.2f}%".format))

    baseline_score = comparison.get("Baseline (Alpha=0)", 0.0)
    attack_score = comparison.get("Attack (Alpha>0)", 0.0)

    delta = attack_score - baseline_score

    print("-" * 70)
    print(
        f"Attack increased jailbreak success by "
        f"{delta:.2f} percentage points."
    )

    # ---------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))

    ax = sns.barplot(
        x=comparison.index,
        y=comparison.values,
        palette=["#2ca02c", "#d62728"]  # Green = baseline, Red = attack
    )

    ax.set_title(
        "Impact of Steering on Jailbreak Success",
        fontsize=16,
        fontweight="bold",
        pad=15
    )
    ax.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)

    for i, v in enumerate(comparison.values):
        ax.text(
            i,
            v + 2,
            f"{v:.1f}%",
            ha="center",
            fontweight="bold",
            fontsize=14
        )

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close()

    print(f"--> Saved plot to {output_plot}")

print("\n✅ All analyses completed successfully.")
