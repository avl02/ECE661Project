import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIG ──────────────────────────────────────────────────────────────────────
LSTM_PATH    = 'results/experiment_openbb_100assets_lstm_cpnone_len63_notime_div_v1'
CPD_PATH     = 'results/experiment_openbb_100assets_lstm_cp21_len63_notime_div_v1'
CPD_SCORE_CSV= 'data/openbb_cpd_21lbw.csv'
START_DATE   = '2007-06-01'
END_DATE     = '2009-06-01'
OUTFILE      = 'visuals/final/plots/scatter_cpd_alpha.png'

# ── HELPERS ─────────────────────────────────────────────────────────────────────
def load_and_pipeline(path):
    # load raw returns
    parts = []
    for sub in sorted(os.listdir(path)):
        f = os.path.join(path, sub, 'captured_returns_fw.csv')
        if not os.path.exists(f): continue
        df = pd.read_csv(f, parse_dates=['time'])
        df.set_index('time', inplace=True)
        parts.append(df['captured_returns'])
    raw = pd.concat(parts).sort_index().loc[START_DATE:END_DATE]
    # use constant 1.0 signal to vol-scale
    sig = pd.Series(1.0, index=raw.index)
    # apply lean_pipeline from previous script
    # replicate here:
    SMOOTH_W, VOL_W, TARGET_VOL = 5, 21, 0.10
    s = sig.rolling(SMOOTH_W).mean().fillna(0)
    pos_pre = s.shift(1)
    vol    = (pos_pre * raw).rolling(VOL_W).std().fillna(method='bfill')
    scale  = TARGET_VOL/np.sqrt(252)
    pos    = (s * (scale/vol)).shift(1).fillna(0)
    return pos * raw

def load_cpd_score_monthly(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.set_index('Date').loc[START_DATE:END_DATE]
    avg = df.pivot(columns='ticker', values='cp_score_21').mean(axis=1)
    # monthly average
    return avg.resample('M').mean()

# ── MAIN ─────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

    # 1) compute monthly returns
    ret_nl  = load_and_pipeline(LSTM_PATH)
    ret_cpd = load_and_pipeline(CPD_PATH)
    m_nl    = (1+ret_nl).resample('M').prod() - 1
    m_cpd   = (1+ret_cpd).resample('M').prod() - 1

    # 2) monthly CPD score
    score_m = load_cpd_score_monthly(CPD_SCORE_CSV)

    # 3) scatter
    plt.figure(figsize=(8,8))
    sc = plt.scatter(
        m_nl, m_cpd,
        c=score_m.loc[m_nl.index],
        cmap='coolwarm', s=60, edgecolor='k'
    )
    lims = [
        min(m_nl.min(), m_cpd.min()),
        max(m_nl.max(), m_cpd.max())
    ]
    plt.plot(lims, lims, 'k--', alpha=0.7)
    plt.colorbar(sc, label='Avg. Monthly CPD Score')
    plt.xlabel('Plain LSTM Monthly Return')
    plt.ylabel('CPD‑LSTM Monthly Return')
    plt.title('CPD Alpha vs. Plain LSTM (Jun 2008 – Jan 2009)')
    plt.tight_layout()
    plt.savefig(OUTFILE, dpi=150)
    print(f"Saved {OUTFILE}")

if __name__ == '__main__':
    main()
