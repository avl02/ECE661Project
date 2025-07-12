#!/usr/bin/env python3
# visualize_performance_monthly.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIG ──────────────────────────────────────────────────────────────────────
STRATEGIES = {
    'No CPD (LSTM)':   'results/experiment_openbb_100assets_lstm_cpnone_len63_notime_div_v1',
    'CPD 21‑day (LSTM)': 'results/experiment_openbb_100assets_lstm_cp21_len63_notime_div_v1',
}
PRICE_CSV  = 'data/openbb_cpd_nonelbw.csv'
START_DATE = '2007-01-01'
END_DATE   = '2009-12-31'
VISUALS_DIR = 'visuals'

# ── HELPERS ─────────────────────────────────────────────────────────────────────
def load_daily_returns(folder):
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
        raise FileNotFoundError(f'No captured_returns_fw.csv under {folder}')
    return pd.concat(parts).sort_index().loc[START_DATE:END_DATE]

def build_true_buy_hold_daily(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    price = df.pivot(index='Date', columns='ticker', values='close')
    price = price.sort_index().loc[START_DATE:END_DATE]
    normed = price.div(price.iloc[0])
    port = normed.mean(axis=1)
    daily_ret = port.pct_change().fillna(0)
    return daily_ret

def compute_monthly_metrics(daily_ret):
    # 1) monthly absolute return
    m_ret = (1 + daily_ret).resample('M').prod() - 1
    # 2) monthly Sharpe (annualized via √252)
    def sharpe(x):
        return x.mean()/x.std()*np.sqrt(252) if x.std()>0 else 0.0
    m_shp = daily_ret.resample('M').apply(sharpe)
    # 3) monthly max drawdown
    def max_dd(x):
        eq = (1 + x).cumprod()
        return (eq/eq.cummax() - 1).min()
    m_dd = daily_ret.resample('M').apply(max_dd)
    # trim to window
    idx = m_ret.index
    mask = (idx>=pd.to_datetime(START_DATE)) & (idx<=pd.to_datetime(END_DATE))
    return m_ret[mask], m_shp[mask], m_dd[mask]

# ── MAIN ─────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(VISUALS_DIR, exist_ok=True)

    # load daily returns for each LSTM strategy
    metrics = {}
    for name, path in STRATEGIES.items():
        daily = load_daily_returns(path)
        metrics[name] = compute_monthly_metrics(daily)

    # true un-rebalanced buy & hold
    bh_daily = build_true_buy_hold_daily(PRICE_CSV)
    metrics['Buy & Hold'] = compute_monthly_metrics(bh_daily)

    months = next(iter(metrics.values()))[0].index

    # 1) Monthly Absolute Return
    plt.figure(figsize=(10,5))
    for name, (m_ret, _, _) in metrics.items():
        plt.plot(months, m_ret, marker='o', label=name)
    plt.title('Monthly Absolute Return (2007–2009)')
    plt.ylabel('Return')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{VISUALS_DIR}/monthly_abs_return.png', dpi=150)

    # 2) Monthly Sharpe Ratio
    plt.figure(figsize=(10,5))
    for name, (_, m_shp, _) in metrics.items():
        plt.plot(months, m_shp, marker='o', label=name)
    plt.title('Monthly Sharpe Ratio (2007–2009)')
    plt.ylabel('Sharpe')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{VISUALS_DIR}/monthly_sharpe.png', dpi=150)

    # 3) Monthly Max Drawdown
    plt.figure(figsize=(10,5))
    for name, (_, _, m_dd) in metrics.items():
        plt.plot(months, m_dd, marker='o', label=name)
    plt.title('Monthly Max Drawdown (2007–2009)')
    plt.ylabel('Drawdown')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{VISUALS_DIR}/monthly_drawdown.png', dpi=150)

    print("Saved 3 monthly plots to 'visuals/opt_monthly_*.png'")

if __name__ == '__main__':
    main()
