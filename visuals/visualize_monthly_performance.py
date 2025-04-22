#!/usr/bin/env python3
# visualize_monthly_performance.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIG ──────────────────────────────────────────────────────────────────────
STRATEGIES = {
    'No CPD (LSTM)':     'results/experiment_openbb_100assets_lstm_cpnone_len63_notime_div_v1',
    'CPD 21-day (LSTM)': 'results/experiment_openbb_100assets_lstm_cp21_len63_notime_div_v1',
}
PRICE_CSV  = 'data/openbb_cpd_nonelbw.csv'
START_DATE = '2007-01-01'
END_DATE   = '2009-12-31'

# ── HELPERS ─────────────────────────────────────────────────────────────────────
def load_daily_returns(folder: str) -> pd.Series:
    parts = []
    for sub in sorted(os.listdir(folder)):
        f = os.path.join(folder, sub, 'captured_returns_fw.csv')
        if os.path.isfile(f):
            df = pd.read_csv(f)
            tcol = next(c for c in df.columns if 'time' in c.lower())
            df[tcol] = pd.to_datetime(df[tcol])
            parts.append(df.set_index(tcol)['captured_returns'])
    if not parts:
        raise FileNotFoundError(f'No captured_returns_fw.csv under {folder}')
    all_daily = pd.concat(parts).sort_index()
    return all_daily.loc[START_DATE:END_DATE]

def build_benchmark_daily(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    price = df.pivot(index='Date', columns='ticker', values='close')
    daily = price.pct_change().mean(axis=1)
    return daily.loc[START_DATE:END_DATE]

def compute_monthly_metrics(daily_returns: pd.Series):
    # 1) Monthly absolute return
    monthly_ret = (1 + daily_returns).resample('M').prod() - 1

    # 2) Monthly Sharpe (annualized via sqrt(252))
    def sharpe_of(x):
        return (x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else np.nan
    monthly_sharpe = daily_returns.resample('M').apply(sharpe_of)

    # 3) Monthly max drawdown
    def max_dd(x):
        eq = (1 + x).cumprod()
        return (eq / eq.cummax() - 1).min()
    monthly_max_dd = daily_returns.resample('M').apply(max_dd)

    # Clip to analysis window
    idx = monthly_ret.index
    mask = (idx >= START_DATE) & (idx <= END_DATE)
    return (
        monthly_ret[mask],
        monthly_sharpe[mask],
        monthly_max_dd[mask]
    )

# ── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    metrics = {}

    # Strategies
    for name, path in STRATEGIES.items():
        daily = load_daily_returns(path)
        metrics[name] = compute_monthly_metrics(daily)

    # Benchmark
    bench_daily = build_benchmark_daily(PRICE_CSV)
    metrics['Buy & Hold'] = compute_monthly_metrics(bench_daily)

    # Unpack for plotting
    dates = next(iter(metrics.values()))[0].index
    for key in ['monthly_absolute_return', 'monthly_sharpe', 'monthly_max_drawdown']:
        plt.figure(figsize=(10,5))
        for name, (mret, mshr, mdd) in metrics.items():
            if key == 'monthly_absolute_return':
                plt.plot(dates, mret.values, marker='o', label=name)
                plt.ylabel('Return')
                plt.title('Monthly Absolute Return (2007–2009)')
            elif key == 'monthly_sharpe':
                plt.plot(dates, mshr.values, marker='o', label=name)
                plt.ylabel('Sharpe')
                plt.title('Monthly Sharpe Ratio (2007–2009)')
            else:  # monthly_max_drawdown
                plt.plot(dates, mdd.values, marker='o', label=name)
                plt.ylabel('Drawdown')
                plt.title('Monthly Max Drawdown (2007–2009)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'visuals/{key}.png', dpi=150)
        print(f'Saved {key}.png')

if __name__ == '__main__':
    main()
