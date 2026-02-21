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
sns.set_theme(style="whitegrid", context="talk")

csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

print(f"--> Found {len(csv_files)} evaluated CSV files")

# ==========================================
# PROCESS EACH FILE
# ==========================================
for csv_file in csv_files:
    input_path = os.path.join(INPUT_DIR, csv_file)
    base_name = csv_file.replace(".csv", "")
    output_image = os.path.join(
        OUTPUT_DIR,
        f"alpha_vs_success_correlation_{base_name}.png"
    )

    print("\n" + "=" * 70)
    print(f" ANALYZING FILE: {csv_file}")
    print("=" * 70)

    df = pd.read_csv(input_path)

    # ==========================================
    # 1. NORMALIZE SUCCESS COLUMN
    # ==========================================
    if df["Jailbreak_Success"].dtype == "object":
        df["Success_Numeric"] = df["Jailbreak_Success"].apply(
            lambda x: 1 if str(x).lower() == "true" else 0
        )
    else:
        df["Success_Numeric"] = df["Jailbreak_Success"].astype(int)

    # ==========================================
    # 2. SUCCESS RATE PER ALPHA
    # ==========================================
    alpha_stats = (
        df.groupby("Alpha")["Success_Numeric"]
        .mean()
        * 100
    ).reset_index()

    print("\nSuccess Rate by Alpha:")
    print(alpha_stats.to_string(index=False))

    # ==========================================
    # 3. PLOT
    # ==========================================
    plt.figure(figsize=(10, 6))

    ax = sns.lineplot(
        data=alpha_stats,
        x="Alpha",
        y="Success_Numeric",
        marker="o",
        markersize=12,
        linewidth=3,
        color="#d62728"  # Red = attack success
    )

    for x, y in zip(alpha_stats["Alpha"], alpha_stats["Success_Numeric"]):
        ax.text(
            x,
            y + 4,
            f"{y:.0f}%",
            ha="center",
            fontsize=12,
            fontweight="bold"
        )

    ax.set_title(
        "Correlation: Steering Strength (Alpha) vs. Jailbreak Success",
        fontweight="bold",
        pad=20
    )
    ax.set_xlabel("Alpha (Steering Intensity)", fontweight="bold")
    ax.set_ylabel("Success Rate (%)", fontweight="bold")
    ax.set_ylim(-5, 115)
    ax.set_xticks(alpha_stats["Alpha"])

    plt.fill_between(
        alpha_stats["Alpha"],
        alpha_stats["Success_Numeric"],
        alpha=0.1,
        color="#d62728"
    )

    # ==========================================
    # 4. SAVE
    # ==========================================
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.close()

    print(f"--> Saved plot to {output_image}")

print("\n✅ Alpha-vs-success analysis completed for all files.")
