import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── PATH CONFIG ─────────────────────────────────────────────────────────────────
# Determine project root based on this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
DATA_DIR    = os.path.join(PROJECT_ROOT, 'data')
VISUALS_DIR = os.path.join(PROJECT_ROOT, 'visuals', 'final', 'plots')

BASELINE_PATH = os.path.join(
    RESULTS_DIR, 'experiment_openbb_100assets_lstm_cpnone_len63_notime_div_v1'
)
CPD21_PATH = os.path.join(
    RESULTS_DIR, 'experiment_openbb_100assets_lstm_cp21_len63_notime_div_v1'
)
CPD_SCORE_CSV = os.path.join(DATA_DIR, 'openbb_cpd_21lbw.csv')

# ── ANALYSIS CONFIG ─────────────────────────────────────────────────────────────
START_DATE      = '2007-08-01'
END_DATE        = '2009-03-31'
WINDOW_DAYS     = 10    # days before/after event
THRESHOLD_PCTL  = 0.95  # top 5% CPD scores

SMOOTH_WINDOW   = 5
VOL_WINDOW      = 21
TARGET_VOL      = 0.10

OUTFILE = os.path.join(VISUALS_DIR, 'event_study_pnl.png')

# ── HELPERS ─────────────────────────────────────────────────────────────────────
def load_strategy_returns(path):
    """Concatenate captured_returns_fw.csv from each subfolder into a daily-return Series."""
    parts = []
    for sub in sorted(os.listdir(path)):
        f = os.path.join(path, sub, 'captured_returns_fw.csv')
        if not os.path.isfile(f):
            continue
        df = pd.read_csv(f, parse_dates=['time'])
        df.set_index('time', inplace=True)
        parts.append(df['captured_returns'])
    sr = pd.concat(parts).sort_index()
    return sr.loc[START_DATE:END_DATE]

def lean_pipeline(signal, returns):
    """Smooth + vol scale + shift to produce enhanced daily returns."""
    s       = signal.rolling(SMOOTH_WINDOW).mean().fillna(0)
    pos_pre = s.shift(1)
    vol     = (pos_pre * returns).rolling(VOL_WINDOW).std().fillna(method='bfill')
    scale   = TARGET_VOL / np.sqrt(252)
    pos     = (s * (scale / vol)).shift(1).fillna(0)
    return pos * returns

def load_cpd_events():
    """Return sorted event dates where avg cp_score_21 >= THRESHOLD_PCTL percentile."""
    df = pd.read_csv(CPD_SCORE_CSV, parse_dates=['Date'])
    df = df[(df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)]
    avg = df.groupby('Date')['cp_score_21'].mean()
    thresh = avg.quantile(THRESHOLD_PCTL)
    return avg[avg >= thresh].index.sort_values()

# ── MAIN ─────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(VISUALS_DIR, exist_ok=True)

    # 1) Load raw returns
    ret_nl_raw  = load_strategy_returns(BASELINE_PATH)
    ret_c21_raw = load_strategy_returns(CPD21_PATH)

    # 2) Create constant 1.0 signals for vol‑scaling
    sig_nl  = pd.Series(1.0, index=ret_nl_raw.index)
    sig_c21 = pd.Series(1.0, index=ret_c21_raw.index)

    # 3) Enhance returns
    daily_nl   = lean_pipeline(sig_nl,  ret_nl_raw)
    daily_cpd  = lean_pipeline(sig_c21, ret_c21_raw)

    # 4) Filter events with full ±WINDOW_DAYS coverage
    events_all = load_cpd_events()
    dates = daily_nl.index
    valid_events = []
    for e in events_all:
    # find all integer indices where the date matches
        locs = np.where(dates == e)[0]
        if len(locs) > 0:
            pos = locs[0]           # take the first match
            # ensure we have a full ±WINDOW_DAYS around it
            if pos >= WINDOW_DAYS and (pos + WINDOW_DAYS) < len(dates):
                valid_events.append(e)
    events = pd.DatetimeIndex(valid_events)

    # 5) Collect event‑aligned P&L
    idx = np.arange(-WINDOW_DAYS, WINDOW_DAYS+1)
    pnl_nl  = pd.DataFrame(index=idx)
    pnl_cpd = pd.DataFrame(index=idx)
    for e in events:
        pos = dates.get_loc(e)
        window = slice(pos-WINDOW_DAYS, pos+WINDOW_DAYS+1)
        window_nl  = daily_nl.iloc[window]
        window_cpd = daily_cpd.iloc[window]
        eq_nl  = (1 + window_nl).cumprod()
        eq_cpd = (1 + window_cpd).cumprod()
        rel_nl  = eq_nl  / eq_nl.iloc[WINDOW_DAYS] - 1
        rel_cpd = eq_cpd / eq_cpd.iloc[WINDOW_DAYS] - 1
        rel_nl.index  = idx
        rel_cpd.index = idx
        pnl_nl[e]  = rel_nl.values
        pnl_cpd[e] = rel_cpd.values

    # 6) Average across events
    avg_nl  = pnl_nl.mean(axis=1)
    avg_cpd = pnl_cpd.mean(axis=1)

    # 7) Plot
    plt.figure(figsize=(10,6))
    plt.plot(idx, avg_nl,  marker='o', label='LSTM')
    plt.plot(idx, avg_cpd, marker='o', label='LSTM + CPD')
    plt.axvline(0, color='black', linestyle='--', alpha=0.7)
    plt.title(f'Average ±{WINDOW_DAYS}-Day P&L around top {(1-THRESHOLD_PCTL)*100:.0f}% CPD events\n'
              f'{START_DATE} to {END_DATE}')
    plt.xlabel('Days relative to event')
    plt.ylabel('Cumulative P&L')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTFILE, dpi=150)
    print(f'Saved event study plot to {OUTFILE}')

if __name__ == '__main__':
    main()