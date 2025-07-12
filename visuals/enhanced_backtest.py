import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIG ──────────────────────────────────────────────────────────────────────
BASELINE_PATH = 'results/experiment_openbb_100assets_lstm_cpnone_len63_notime_div_v1'
CPD21_PATH    = 'results/experiment_openbb_100assets_lstm_cp21_len63_notime_div_v1'
PRICE_CSV     = 'data/openbb_cpd_nonelbw.csv'

START_DATE    = '2007-01-01'
END_DATE      = '2009-12-31'

SMOOTH_WINDOW = 5       # days for moving-average smoothing
VOL_WINDOW    = 21      # days for rolling volatility
TARGET_VOL    = 0.10    # annualized target volatility (10%)

VISUALS_DIR   = 'visuals'

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

def build_benchmark_daily(csv_path):
    """
    True un‑rebalanced buy & hold:
    - pivot close prices
    - normalize each series to 1 at START_DATE
    - average across tickers to get portfolio value
    - compute daily pct change
    """
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    price = df.pivot(index='Date', columns='ticker', values='close')
    # restrict window
    price = price.loc[START_DATE:END_DATE]
    # normalize each to 1 at first available date
    normed = price.div(price.iloc[0])
    # equal‑weight portfolio value
    port = normed.mean(axis=1)
    # daily returns
    daily_ret = port.pct_change().fillna(0)
    return daily_ret

def lean_pipeline(signal, returns):
    # 1. Smooth raw position signal
    s = signal.rolling(SMOOTH_WINDOW).mean().fillna(0)
    # 2. Prepare for volatility estimate
    pos_pre = s.shift(1)
    daily_pre = pos_pre * returns
    # 3. Rolling volatility
    vol = daily_pre.rolling(VOL_WINDOW).std().fillna(method='bfill')
    # 4. Scale to target volatility
    scale = TARGET_VOL / np.sqrt(252)
    pos_scaled = s * (scale / vol)
    # 5. Execute next day
    pos_exec = pos_scaled.shift(1).fillna(0)
    daily_ret = pos_exec * returns
    return daily_ret

def compute_monthly_metrics(daily_ret):
    # Monthly absolute return
    abs_ret = (1 + daily_ret).resample('M').prod() - 1

    # Monthly Sharpe (annualized via sqrt(252)), fill NaN with 0
    sharpe = daily_ret.resample('M').apply(
        lambda x: x.mean()/x.std()*np.sqrt(252) if x.std()>0 else 0.0
    )

    # Monthly max drawdown
    def month_max_dd(x):
        eq = (1 + x).cumprod()
        return (eq/eq.cummax() - 1).min()
    max_dd = daily_ret.resample('M').apply(month_max_dd)

    # Trim to window
    mask = (abs_ret.index >= pd.to_datetime(START_DATE)) & \
           (abs_ret.index <= pd.to_datetime(END_DATE))
    return abs_ret[mask], sharpe[mask], max_dd[mask]

# ── MAIN ─────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(VISUALS_DIR, exist_ok=True)

    # Load model signals & returns
    sig_nl, ret_nl = load_strategy(BASELINE_PATH)
    sig_21, ret_21 = load_strategy(CPD21_PATH)
    # True buy & hold daily returns
    ret_bh = build_benchmark_daily(PRICE_CSV)

    # Apply the 3-step pipeline (smoothing + vol scaling)
    daily_nl = lean_pipeline(sig_nl, ret_nl)
    daily_21 = lean_pipeline(sig_21,  ret_21)
    daily_bh = ret_bh

    # Compute monthly metrics
    names = ['Buy & Hold', 'LSTM No Lookback', 'LSTM 21-day CPD']
    series = [daily_bh, daily_nl, daily_21]
    monthly = {}
    for name, daily in zip(names, series):
        abs_r, shp, dd = compute_monthly_metrics(daily)
        monthly[name] = {'abs': abs_r, 'shp': shp, 'dd': dd}

    months = monthly[names[0]]['abs'].index

    # 1) Monthly Absolute Return
    plt.figure(figsize=(10,5))
    for name in names:
        plt.plot(months, monthly[name]['abs'], marker='o', label=name)
    plt.title('Monthly Absolute Return (2007–2009)')
    plt.ylabel('Return')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{VISUALS_DIR}/opt_monthly_abs_return.png", dpi=150)

    # 2) Monthly Sharpe Ratio
    plt.figure(figsize=(10,5))
    for name in names:
        plt.plot(months, monthly[name]['shp'], marker='o', label=name)
    plt.title('Monthly Sharpe Ratio (2007–2009)')
    plt.ylabel('Sharpe')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{VISUALS_DIR}/opt_monthly_sharpe.png", dpi=150)

    # 3) Monthly Max Drawdown
    plt.figure(figsize=(10,5))
    for name in names:
        plt.plot(months, monthly[name]['dd'], marker='o', label=name)
    plt.title('Monthly Max Drawdown (2007–2009)')
    plt.ylabel('Drawdown')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{VISUALS_DIR}/opt_monthly_max_drawdown.png", dpi=150)

    print("Saved 3 monthly charts under 'visuals/opt_*.png'")

if __name__ == '__main__':
    main()