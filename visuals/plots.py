#!/usr/bin/env python
import os
import pandas as pd
import matplotlib.pyplot as plt

def main(
    csv_path: str = "data/openbb_cpd_21lbw.csv",
    output_dir: str = "visuals/plots"
):
    # 1) load
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)

    os.makedirs(output_dir, exist_ok=True)

    # 2) plot changepoint score
    plt.figure()
    df["cp_score_21"].plot()
    plt.title("Changepoint Severity Score (21‑day LBW)")
    plt.xlabel("Date")
    plt.ylabel("Severity Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cp_score_21.png"))
    plt.close()

    # 3) plot captured returns
    plt.figure()
    df["cp_rl_21"].plot()
    plt.title("Captured Returns (21‑day LBW)")
    plt.xlabel("Date")
    plt.ylabel("Captured Returns")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cp_rl_21.png"))
    plt.close()

    # 4) scatter score vs returns
    plt.figure()
    df.plot.scatter(x="cp_score_21", y="cp_rl_21")
    plt.title("Changepoint Score vs Captured Returns")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_vs_returns.png"))
    plt.close()

    print(f"All plots saved to '{output_dir}'.")

if __name__ == "__main__":
    main()
