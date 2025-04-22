import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIG ──────────────────────────────────────────────────────────────────────
LSTM_PATH     = 'results/experiment_openbb_100assets_lstm_cpnone_len63_notime_div_v1'
CPD_PATH      = 'results/experiment_openbb_100assets_lstm_cp21_len63_notime_div_v1'
CPD_SCORE_CSV = 'data/openbb_cpd_21lbw.csv'  # must contain Date and cp_score_21 columns
START_DATE    = '2008-06-01'
END_DATE      = '2009-01-01'
SMOOTH_W      = 5
VOL_W         = 21
TARGET_VOL    = 0.10
OUTFILE       = 'visuals/final/plots/equity_cpd_overlay.png'

# ── HELPERS ─────────────────────────────────────────────────────────────────────
def load_strategy_returns(path):
    parts = []
    for sub in sorted(os.listdir(path)):
        f = os.path.join(path, sub, 'captured_returns_fw.csv')
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, parse_dates=['time'])
        df.set_index('time', inplace=True)
        parts.append(df['captured_returns'])
    all_ret = pd.concat(parts).sort_index()
    return all_ret.loc[START_DATE:END_DATE]

def lean_pipeline(signal, returns):
    s       = signal.rolling(SMOOTH_W).mean().fillna(0)
    pos_pre = s.shift(1)
    vol     = (pos_pre * returns).rolling(VOL_W).std().fillna(method='bfill')
    scale   = TARGET_VOL / np.sqrt(252)
    pos     = (s * (scale / vol)).shift(1).fillna(0)
    return pos * returns

def load_cpd_events(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    # filter to date window
    df = df[(df['Date'] >= START_DATE) & (df['Date'] < END_DATE)]
    # average CPD score across tickers per day
    avg = df.groupby('Date')['cp_score_21'].mean()
    # pick top 5% days
    thresh = avg.quantile(0.95)
    events = avg[avg >= thresh].index
    return events.sort_values()

# ── MAIN ─────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

    # 1) raw returns
    ret_nl_raw = load_strategy_returns(LSTM_PATH)
    ret_cpd_raw= load_strategy_returns(CPD_PATH)

    # 2) constant signal = 1.0 for vol-scaling
    sig_nl = pd.Series(1.0, index=ret_nl_raw.index)
    sig_cpd= pd.Series(1.0, index=ret_cpd_raw.index)

    # 3) apply pipeline
    daily_nl  = lean_pipeline(sig_nl,  ret_nl_raw)
    daily_cpd = lean_pipeline(sig_cpd, ret_cpd_raw)

    # 4) cumulative curves
    cum_nl  = (1 + daily_nl).cumprod()
    cum_cpd = (1 + daily_cpd).cumprod()

    # 5) CPD event dates
    events = load_cpd_events(CPD_SCORE_CSV)

    # 6) plot
    plt.figure(figsize=(12,6))
    plt.plot(cum_nl.index, cum_nl, label='LSTM')
    plt.plot(cum_cpd.index, cum_cpd, label='LSTM + CPD')
    for d in events:
        plt.axvline(d, color='gray', alpha=0.3)
    plt.title('Daily Equity with CPD Events (Jun 2008 – Jan 2009)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTFILE, dpi=150)
    print(f"Saved {OUTFILE}")

if __name__ == '__main__':
    main()