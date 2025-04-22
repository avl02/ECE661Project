#!/usr/bin/env python3
# visualize_performance.py
#
#  • Total Return bar chart         → total_return_comparison.png
#  • Sharpe Ratio bar chart         → sharpe_comparison.png
#  • Draw‑down line plot            → drawdown_comparison.png
#
# Adjust STRATEGIES paths or PRICE_CSV if your folders differ.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIG ──────────────────────────────────────────────────────────────────────
STRATEGIES = {
    'No CPD (LSTM)':        'results/experiment_openbb_100assets_lstm_cpnone_len63_notime_div_v1',
    'CPD 21‑day (LSTM)':    'results/experiment_openbb_100assets_lstm_cp21_len63_notime_div_v1',
}

# equal‑weight buy‑&‑hold price file
PRICE_CSV  = 'data/openbb_cpd_nonelbw.csv'
START_DATE = '2007-01-01'
END_DATE   = '2009-12-31'

# ── HELPERS ─────────────────────────────────────────────────────────────────────
def load_daily_returns(folder: str) -> pd.Series:
    """Concatenate all captured_returns_fw.csv files in *folder* into one daily‑return Series."""
    parts = []
    for sub in sorted(os.listdir(folder)):
        f = os.path.join(folder, sub, 'captured_returns_fw.csv')
        if not os.path.isfile(f):
            continue
        df = pd.read_csv(f)
        tcol = next(c for c in df.columns if 'time' in c.lower())
        df[tcol] = pd.to_datetime(df[tcol])
        parts.append(df.set_index(tcol)['captured_returns'])
    if not parts:
        raise FileNotFoundError(f'No captured_returns_fw.csv found under {folder}')
    return pd.concat(parts).sort_index().loc[START_DATE:END_DATE]

def monthify(ser: pd.Series):
    m = (1 + ser).resample('M').prod() - 1
    eq = (1 + m).cumprod()
    return m, eq

def buy_hold(csv: str):
    df = pd.read_csv(csv, parse_dates=['Date'])
    px = df.pivot(index='Date', columns='ticker', values='close')
    mpx = px.resample('M').last().loc[START_DATE:END_DATE]
    mret = mpx.pct_change().mean(axis=1)
    eq   = (1 + mret).cumprod()
    return mret, eq

def stats(m: pd.Series, eq: pd.Series):
    total  = eq.iloc[-1] - 1
    sharpe = m.mean() / m.std() * np.sqrt(12)
    dd     = eq / eq.cummax() - 1
    return total, sharpe, dd

# ── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    totals, sharpes, dds = {}, {}, {}

    # strategies
    for name, path in STRATEGIES.items():
        m, eq = monthify(load_daily_returns(path))
        tot, sh, dd = stats(m, eq)
        totals[name], sharpes[name], dds[name] = tot, sh, dd

    # buy‑&‑hold benchmark
    m, eq = buy_hold(PRICE_CSV)
    tot, sh, dd = stats(m, eq)
    totals['Buy & Hold'], sharpes['Buy & Hold'], dds['Buy & Hold'] = tot, sh, dd

    # --- Total Return bar chart -------------------------------------------------
    plt.figure(figsize=(8,5))
    plt.bar(totals.keys(), totals.values(), color='tab:blue')
    plt.xticks(rotation=15)
    plt.ylabel('Total Return')
    plt.title('Absolute Return (2007‑09)')
    plt.tight_layout()
    plt.savefig('visuals/total_return_comparison.png', dpi=150)

    # --- Sharpe bar chart -------------------------------------------------------
    plt.figure(figsize=(8,5))
    plt.bar(sharpes.keys(), sharpes.values(), color='tab:orange')
    plt.xticks(rotation=15)
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio (annualised, 2007‑09)')
    plt.tight_layout()
    plt.savefig('visuals/sharpe_comparison.png', dpi=150)

    # --- Draw‑down line plot ----------------------------------------------------
    plt.figure(figsize=(10,5))
    for name, dd in dds.items():
        plt.plot(dd.index, dd.values, label=name)
    plt.legend()
    plt.ylabel('Draw‑down')
    plt.title('Draw‑down Comparison (2007‑09)')
    plt.tight_layout()
    plt.savefig('visuals/drawdown_comparison.png', dpi=150)


if __name__ == '__main__':
    main()
