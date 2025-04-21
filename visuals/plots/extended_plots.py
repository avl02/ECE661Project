#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(
    csv_path: str = "data/openbb_cpd_21lbw.csv",
    output_dir: str = "visuals/extended_plots",
):
    # load & prepare
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)
    df["Year"] = df.index.year
    df["Month"] = df.index.month

    os.makedirs(output_dir, exist_ok=True)

    # 1) Histogram of changepoint severity
    plt.figure(figsize=(6,4))
    vals = df["cp_score_21"].dropna()
    plt.hist(vals, bins=40, edgecolor="black")
    plt.title("Histogram of Changepoint Severity (21‑day LBW)")
    plt.xlabel("Severity Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hist_cp_score.png"))
    plt.close()

    # 2) Boxplot of severity by year
    years = sorted(df["Year"].unique())
    data_by_year = [df.loc[df["Year"]==y, "cp_score_21"].dropna() for y in years]
    plt.figure(figsize=(8,5))
    plt.boxplot(data_by_year, labels=years, showfliers=False)
    plt.xticks(rotation=45)
    plt.title("Yearly Distribution of Severity Score")
    plt.xlabel("Year")
    plt.ylabel("Severity Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplot_cp_score_by_year.png"))
    plt.close()

    # 3) Rolling‐30‑day averages of score and captured returns
    roll = df[["cp_score_21","cp_rl_21"]].rolling(window=30, min_periods=1).mean()
    plt.figure(figsize=(8,4))
    plt.plot(roll.index, roll["cp_score_21"], label="Score (30‑day MA)")
    plt.plot(roll.index, roll["cp_rl_21"],    label="Captured Returns (30‑day MA)")
    plt.legend()
    plt.title("Rolling 30‑day Averages")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rolling30_cp_score_vs_returns.png"))
    plt.close()

    # 4) Scatter Score vs Returns colored by time
    # map date to ordinal for color
    ordinals = df.index.map(pd.Timestamp.toordinal)
    plt.figure(figsize=(6,6))
    sc = plt.scatter(
        df["cp_score_21"], df["cp_rl_21"],
        c=ordinals, cmap="viridis", s=10, alpha=0.6
    )
    plt.colorbar(sc, label="Date")
    plt.title("Score vs Captured Returns (colored by time)")
    plt.xlabel("Severity Score")
    plt.ylabel("Captured Returns")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_score_vs_returns_time.png"))
    plt.close()

    # 5) Heatmap of average severity by Year×Month
    pivot = df.pivot_table(
        index="Year", columns="Month", values="cp_score_21", aggfunc="mean"
    )
    plt.figure(figsize=(8,6))
    plt.imshow(pivot, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(label="Avg Severity Score")
    plt.xticks(np.arange(0,12), pivot.columns, rotation=0)
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.title("Heatmap: Avg Severity by Year & Month")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_score_year_month.png"))
    plt.close()

    print(f"Saved 5 plots to '{output_dir}'")

if __name__ == "__main__":
    main()
