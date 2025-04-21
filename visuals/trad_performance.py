#!/usr/bin/env python3
# trad_performance.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── CONFIG ─────────────────────────────────────────────────────────────────────

# Map strategy names to folders containing year subfolders with captured_returns_fw.csv
STRATEGIES = {
    'CPD (21d)':   'results/experiment_openbb_100assets_lstm_cp21_len63_notime_div_v1',
    'No CPD':      'results/experiment_openbb_100assets_lstm_cpnone_len63_notime_div_v1'
}

# Path to OpenBB price CSV for benchmark
PRICE_CSV = 'data/openbb_cpd_21lbw.csv'

# Date range for focusing on the 2008 crisis
START_DATE = '2007-01-01'
END_DATE   = '2009-12-31'

# Risk-free rate for Sharpe calculation
RFR = 0.0

# ─── HELPERS ────────────────────────────────────────────────────────────────────

def load_strategy_returns(root_folder):
    """
    Walk through subfolders of `root_folder`, find `captured_returns_fw.csv`,
    parse the 'time' (or date-like) column, use 'captured_returns' as returns,
    and return a Date-indexed pd.Series clipped to our window.
    """
    frames = []
    for sub in sorted(os.listdir(root_folder)):
        csv_path = os.path.join(root_folder, sub, 'captured_returns_fw.csv')
        if not os.path.isfile(csv_path):
            continue

        df = pd.read_csv(csv_path)
        print(f"→ Inspecting {csv_path}")
        print("   Columns:", df.columns.tolist())

        # 1) detect date/time column
        date_cols = [
            c for c in df.columns
            if any(key in c.lower() for key in ('date', 'time', 'timestamp'))
        ]
        if not date_cols:
            raise ValueError(f"No date-like column in {csv_path}")
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col])

        # 2) pick returns column: prefer 'captured_returns'
        if 'captured_returns' in df.columns:
            ret_col = 'captured_returns'
        elif 'returns' in df.columns:
            ret_col = 'returns'
        else:
            # fallback to last numeric column
            nums = df.select_dtypes(include='number').columns.tolist()
            nums = [c for c in nums if c != date_col]
            if not nums:
                raise ValueError(f"No numeric column for returns in {csv_path}")
            ret_col = nums[-1]

        # 3) keep only date + return
        frames.append(
            df[[date_col, ret_col]]
              .rename(columns={date_col: 'date', ret_col: 'return'})
        )

    if not frames:
        raise FileNotFoundError(f"No captured_returns_fw.csv under {root_folder}")

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.set_index('date').sort_index()
    return all_df['return'].loc[START_DATE:END_DATE]

def compute_monthly(return_series):
    """
    Convert a Date-indexed series of returns into monthly returns and cumulative equity.
    """
    m_ret = (1 + return_series).resample('M').prod() - 1
    m_cum = (1 + m_ret).cumprod()
    return m_ret, m_cum

def build_benchmark(price_csv):
    """
    Build an equal-weight buy-and-hold benchmark from price CSV:
      - pivot ticker/close
      - resample month-end
      - pct_change for monthly returns
      - mean across tickers
      - cumulative equity
    """
    df = pd.read_csv(price_csv, parse_dates=['Date'])
    price = df.pivot(index='Date', columns='ticker', values='close')
    m_price = price.resample('M').last().loc[START_DATE:END_DATE]
    m_ret   = m_price.pct_change().dropna()
    bench_ret = m_ret.mean(axis=1)
    bench_cum = (1 + bench_ret).cumprod()
    return bench_ret, bench_cum

def performance_metrics(monthly_ret, cum_eq):
    """
    Compute CAGR, annualized volatility, Sharpe ratio, max drawdown, and Calmar ratio.
    """
    n = len(monthly_ret)
    total = cum_eq.iloc[-1]
    CAGR = total**(12/n) - 1
    vol_ann = monthly_ret.std() * np.sqrt(12)
    sharpe  = (CAGR - RFR) / vol_ann if vol_ann > 0 else np.nan
    dd = cum_eq / cum_eq.cummax() - 1
    maxDD = dd.min()
    calmar = CAGR / abs(maxDD) if maxDD < 0 else np.nan
    return {'CAGR': CAGR, 'Vol': vol_ann, 'Sharpe': sharpe, 'MaxDD': maxDD, 'Calmar': calmar}

def main():
    stats = {}
    all_monthly = {}
    all_cum     = {}

    # Load each strategy's returns
    for name, path in STRATEGIES.items():
        print(f'→ Loading {name}')
        dr = load_strategy_returns(path)
        m_ret, m_cum = compute_monthly(dr)
        all_monthly[name] = m_ret
        all_cum[name]     = m_cum
        stats[name]       = performance_metrics(m_ret, m_cum)

    # Build buy-and-hold benchmark
    print('→ Building buy-and-hold benchmark')
    bench_ret, bench_cum = build_benchmark(PRICE_CSV)
    all_monthly['Benchmark'] = bench_ret
    all_cum    ['Benchmark'] = bench_cum
    stats['Benchmark']       = performance_metrics(bench_ret, bench_cum)

    # 1) Cumulative returns plot
    plt.figure(figsize=(10,6))
    for n, cum in all_cum.items():
        plt.plot(cum.index, cum.values, label=n)
    plt.title('Cumulative Equity (2007–09)')
    plt.ylabel('Growth of $1')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cumulative_returns.png', dpi=150)

    # 2) Drawdowns plot
    plt.figure(figsize=(10,4))
    for n, cum in all_cum.items():
        dd = cum / cum.cummax() - 1
        plt.plot(dd.index, dd.values, label=n)
    plt.title('Drawdowns (2007–09)')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('drawdowns.png', dpi=150)

    # 3) Metrics bar chart
    dfm = pd.DataFrame(stats).T
    dfm[['CAGR','Vol','Sharpe','MaxDD','Calmar']].plot(
        kind='bar', figsize=(10,5)
    )
    plt.title('Performance Metrics Comparison')
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=150)

    # 4) Boxplot of 2008 monthly returns
    df_2008 = pd.DataFrame({n: m.loc['2008'] for n, m in all_monthly.items()})
    plt.figure(figsize=(8,5))
    df_2008.boxplot()
    plt.title('Monthly Returns Distribution (2008)')
    plt.ylabel('Return')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('boxplot_2008_returns.png', dpi=150)

if __name__ == '__main__':
    main()
