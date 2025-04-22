#!/usr/bin/env python3
# visualize_performance_weekly.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIG ──────────────────────────────────────────────────────────────────────
BASELINE_PATH = 'results/experiment_openbb_100assets_lstm_cpnone_len63_notime_div_v1'
CPD21_PATH    = 'results/experiment_openbb_100assets_lstm_cp21_len63_notime_div_v1'
PRICE_CSV     = 'data/openbb_cpd_nonelbw.csv'

START_DATE    = '2008-06-01'
END_DATE      = '2008-12-31'

SMOOTH_WINDOW = 5       # days for moving-average smoothing
VOL_WINDOW    = 21      # days for rolling volatility
TARGET_VOL    = 0.10    # annualized target volatility (10%)

VISUALS_DIR   = 'visuals/final/plots'

# ── HELPERS ─────────────────────────────────────────────────────────────────────
def load_strategy(path):
    signals, returns = [], []
    for sub in sorted(os.listdir(path)):
        f = os.path.join(path, sub, 'captured_returns_fw.csv')
        if not os.path.isfile(f):
            continue
        df = pd.read_csv(f)
        tcol = next(c for c in df.columns if 'time' in c.lower())
        df[tcol] = pd.to_datetime(df[tcol])
        df.set_index(tcol, inplace=True)
        signals.append(df['position'])
        returns.append(df['captured_returns'])
    sig = pd.concat(signals).sort_index().loc[START_DATE:END_DATE]
    ret = pd.concat(returns).sort_index().loc[START_DATE:END_DATE]
    return sig, ret

def build_true_buy_hold_daily(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    price = df.pivot(index='Date', columns='ticker', values='close')
    price = price.sort_index().loc[START_DATE:END_DATE]
    normed = price.div(price.iloc[0])
    port = normed.mean(axis=1)
    daily_ret = port.pct_change().fillna(0)
    return daily_ret

def lean_pipeline(signal, returns):
    s = signal.rolling(SMOOTH_WINDOW).mean().fillna(0)
    pos_pre = s.shift(1)
    daily_pre = pos_pre * returns
    vol = daily_pre.rolling(VOL_WINDOW).std().fillna(method='bfill')
    scale = TARGET_VOL / np.sqrt(252)
    pos_scaled = s * (scale / vol)
    pos_exec = pos_scaled.shift(1).fillna(0)
    daily_ret = pos_exec * returns
    return daily_ret

def compute_weekly_metrics(daily_ret):
    # Weekly absolute return (week ending Friday)
    weekly_ret = (1 + daily_ret).resample('W-FRI').prod() - 1

    # Weekly Sharpe (annualized via sqrt(52))
    def sharpe(x):
        return x.mean() / x.std() * np.sqrt(52) if x.std() > 0 else 0.0
    weekly_shp = daily_ret.resample('W-FRI').apply(sharpe)

    # Weekly max drawdown
    def week_max_dd(x):
        eq = (1 + x).cumprod()
        return (eq / eq.cummax() - 1).min()
    weekly_dd = daily_ret.resample('W-FRI').apply(week_max_dd)

    mask = (weekly_ret.index >= pd.to_datetime(START_DATE)) & (weekly_ret.index <= pd.to_datetime(END_DATE))
    return weekly_ret[mask], weekly_shp[mask], weekly_dd[mask]

# ── MAIN ─────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(VISUALS_DIR, exist_ok=True)

    # Load strategies
    sig_nl, ret_nl = load_strategy(BASELINE_PATH)
    sig_21, ret_21 = load_strategy(CPD21_PATH)
    ret_bh         = build_true_buy_hold_daily(PRICE_CSV)

    # Apply pipeline
    weekly_nl = lean_pipeline(sig_nl, ret_nl)
    weekly_21 = lean_pipeline(sig_21, ret_21)
    weekly_bh = ret_bh

    # Compute weekly metrics
    names   = ['Buy & Hold', 'LSTM No Lookback', 'LSTM 21-day CPD']
    series  = [weekly_bh, weekly_nl, weekly_21]
    metrics = {}
    for name, daily in zip(names, series):
        w_ret, w_shp, w_dd = compute_weekly_metrics(daily)
        metrics[name] = {'ret': w_ret, 'shp': w_shp, 'dd': w_dd}

    weeks = metrics[names[0]]['ret'].index

    # 1) Weekly Absolute Return
    plt.figure(figsize=(10,5))
    for name in names:
        plt.plot(weeks, metrics[name]['ret'], marker='o', label=name)
    plt.title('Weekly Absolute Return (Jun–Dec 2008)')
    plt.ylabel('Return')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{VISUALS_DIR}/opt_weekly_abs_return.png", dpi=150)

    # 2) Weekly Sharpe Ratio
    plt.figure(figsize=(10,5))
    for name in names:
        plt.plot(weeks, metrics[name]['shp'], marker='o', label=name)
    plt.title('Weekly Sharpe Ratio (Jun–Dec 2008)')
    plt.ylabel('Sharpe')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{VISUALS_DIR}/opt_weekly_sharpe.png", dpi=150)

    # 3) Weekly Max Drawdown
    plt.figure(figsize=(10,5))
    for name in names:
        plt.plot(weeks, metrics[name]['dd'], marker='o', label=name)
    plt.title('Weekly Max Drawdown (Jun–Dec 2008)')
    plt.ylabel('Drawdown')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{VISUALS_DIR}/opt_weekly_max_drawdown.png", dpi=150)

    print("Saved weekly plots under 'visuals/final/plots'.")
    
if __name__ == '__main__':
    main()
