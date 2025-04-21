#!/usr/bin/env python3
"""
plots.py

A collection of functions to visualize how your strategy and CPD signals behaved
around the 2008 financial crisis. To run:

    python plots.py

Make sure you’ve installed at least:
    pandas, numpy, matplotlib, glob2
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_captured_returns(results_root: str) -> pd.DataFrame:
    """
    Read every sliding‐window captured_returns.csv under results_root/*/
    and return a single DataFrame.
    Assumes each file has columns: identifier, time, returns, position, captured_returns
    """
    files = glob.glob(os.path.join(results_root, "*/captured_returns.csv"))
    if not files:
        raise FileNotFoundError(f"No captured_returns.csv found under {results_root}")
    dfs = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["time"])
        df["window"] = os.path.basename(os.path.dirname(f))
        dfs.append(df.rename(columns={"time": "date"}))
    return pd.concat(dfs, ignore_index=True)


def load_features(features_file: str) -> pd.DataFrame:
    """
    Read your openbb_21lbw.csv (with daily_returns, daily_vol, cp_score_21, cp_rl_21, ticker, Date).
    """
    df = pd.read_csv(features_file, parse_dates=["Date"])
    df = df.rename(columns={"Date": "date"})
    return df


def plot_cumulative(df_all: pd.DataFrame, feats: pd.DataFrame):
    """1. Cumulative strategy P&L vs benchmark (avg daily_returns)."""
    strat = df_all.groupby("date")["captured_returns"].sum().sort_index()
    strat_cum = (1 + strat).cumprod() - 1

    bench = feats.groupby("date")["daily_returns"].mean().sort_index()
    bench_cum = (1 + bench).cumprod() - 1

    plt.figure(figsize=(10, 5))
    plt.plot(strat_cum.index, strat_cum.values, label="Strategy")
    plt.plot(bench_cum.index, bench_cum.values, label="Benchmark (avg daily return)")
    plt.title("Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_avg_position(df_all: pd.DataFrame):
    """2. Average absolute position over time."""
    avg_pos = (
        df_all.groupby("date")["position"]
        .apply(lambda x: np.mean(np.abs(x)))
        .sort_index()
    )
    plt.figure(figsize=(10, 4))
    plt.plot(avg_pos.index, avg_pos.values)
    plt.title("Average Absolute Position Over Time")
    plt.xlabel("Date")
    plt.ylabel("Mean |position|")
    plt.tight_layout()
    plt.show()


def plot_severity_heatmap(feats: pd.DataFrame, start="2007-01-01", end="2009-12-31"):
    """3. Heatmap of cp_score_21 per ticker 2007–2009."""
    sub = feats[(feats["date"] >= start) & (feats["date"] <= end)]
    pivot = sub.pivot(index="ticker", columns="date", values="cp_score_21")
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    # show a tick every ~30 days
    xticks = np.linspace(0, pivot.shape[1] - 1, num=12, dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels(pivot.columns[xticks].strftime("%Y-%m-%d"), rotation=90)
    fig.colorbar(im, ax=ax, label="CPD Severity (cp_score_21)")
    ax.set_title("CPD Severity Heatmap (2007–2009)")
    plt.tight_layout()
    plt.show()


def plot_drawdowns(df_all: pd.DataFrame):
    """4. Max drawdown per one‐year test window."""
    dd = {}
    for window, group in df_all.groupby("window"):
        ser = group.sort_values("date").groupby("date")["captured_returns"].sum()
        cum = (1 + ser).cumprod()
        roll_max = cum.cummax()
        drawdown = (cum - roll_max) / roll_max
        dd[window] = drawdown.min()
    dd_s = pd.Series(dd).sort_index()

    plt.figure(figsize=(8, 4))
    dd_s.plot(kind="bar")
    plt.title("Max Drawdown per Test Window")
    plt.xlabel("Window (e.g. 2008-2009)")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.show()


def plot_rolling_corr(feats: pd.DataFrame, window=30):
    """7. 30‐day rolling correlation between cp_score_21 and daily_vol."""
    daily = feats.groupby("date").agg({"cp_score_21": "mean", "daily_vol": "mean"})
    corr = daily["cp_score_21"].rolling(window).corr(daily["daily_vol"])

    plt.figure(figsize=(10, 4))
    plt.plot(corr.index, corr.values)
    plt.title(f"Rolling {window}‐Day Correlation: CP Score vs Realized Vol")
    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # === adjust these two paths to match your setup ===
    RESULTS_DIR = "results/experiment_openbb_100assets_lstm_cp21_len63_notime_div_v1"
    FEATURES_FILE = "data/openbb_21lbw.csv"

    # load
    df_all = load_captured_returns(RESULTS_DIR)
    feats = load_features(FEATURES_FILE)

    # 1
    plot_cumulative(df_all, feats)

    # 2
    plot_avg_position(df_all)

    # 3
    plot_severity_heatmap(feats)

    # 4
    plot_drawdowns(df_all)

    # 7
    plot_rolling_corr(feats, window=30)
