#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(
    csv_path: str = "data/openbb_cpd_21lbw.csv",
    output_dir: str = "visuals/crisis_plots",
):
    # load & prepare
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)
    # define periods
    crisis_start = pd.Timestamp("2007-08-01")
    crisis_end   = pd.Timestamp("2009-03-01")
    def label_period(d):
        if d < crisis_start:
            return "Pre‑crisis"
        elif d <= crisis_end:
            return "Crisis"
        else:
            return "Post‑crisis"
    df["Period"] = df.index.map(label_period)

    os.makedirs(output_dir, exist_ok=True)

    # 1) Time‐series of severity with crisis shaded
    plt.figure(figsize=(10,4))
    plt.plot(df.index, df["cp_score_21"], label="Severity Score")
    plt.axvspan(crisis_start, crisis_end, color="grey", alpha=0.3, label="Crisis period")
    plt.title("Changepoint Severity Around 2008 Crisis")
    plt.xlabel("Date")
    plt.ylabel("Severity Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ts_severity_crisis.png"))
    plt.close()

    # 2) Histogram of severity by period
    plt.figure(figsize=(8,4))
    for period, color in [("Pre‑crisis","blue"),("Crisis","red"),("Post‑crisis","green")]:
        subset = df.loc[df["Period"]==period, "cp_score_21"].dropna()
        plt.hist(subset, bins=30, alpha=0.5, label=period, color=color, edgecolor="black")
    plt.title("Severity Score Distribution by Period")
    plt.xlabel("Severity Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hist_severity_by_period.png"))
    plt.close()

    # 3) Boxplot of severity by period
    data = [df.loc[df["Period"]==p, "cp_score_21"].dropna() for p in ["Pre‑crisis","Crisis","Post‑crisis"]]
    plt.figure(figsize=(6,4))
    plt.boxplot(data, labels=["Pre‑crisis","Crisis","Post‑crisis"], showfliers=False)
    plt.title("Severity Score Boxplot by Period")
    plt.ylabel("Severity Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplot_severity_by_period.png"))
    plt.close()

    # 4) Scatter severity vs captured returns during crisis
    cri = df.loc[df["Period"]=="Crisis"].dropna(subset=["cp_score_21","cp_rl_21"])
    plt.figure(figsize=(6,6))
    plt.scatter(cri["cp_score_21"], cri["cp_rl_21"], s=20, alpha=0.6)
    plt.title("Severity vs Captured Returns (Crisis Period)")
    plt.xlabel("Severity Score")
    plt.ylabel("Captured Returns")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_crisis_score_vs_returns.png"))
    plt.close()

    # 5) Rolling‐30‑day around crisis (zoomed)
    window = 30
    roll = df[["cp_score_21","cp_rl_21"]].rolling(window=window, min_periods=1).mean()
    zoom = roll.loc[crisis_start - pd.Timedelta("90D") : crisis_end + pd.Timedelta("90D")]
    plt.figure(figsize=(10,4))
    plt.plot(zoom.index, zoom["cp_score_21"], label="Severity (30d MA)")
    plt.plot(zoom.index, zoom["cp_rl_21"],    label="Returns (30d MA)")
    plt.axvspan(crisis_start, crisis_end, color="grey", alpha=0.2)
    plt.title("Rolling 30‑day MA (±90 days around Crisis)")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rolling_crisis_zoom.png"))
    plt.close()

    print(f"Saved crisis‐focused plots to '{output_dir}'")

if __name__ == "__main__":
    main()
