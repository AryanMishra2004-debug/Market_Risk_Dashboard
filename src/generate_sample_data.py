"""
Generates a realistic synthetic daily price dataset for:
RELIANCE.NS, HDFCBANK.NS, TCS.NS, INFY.NS, ^NSEI

This exists as an OFFLINE FALLBACK. The primary path (see data_loader.py)
always tries yfinance first for real historical data. This script is only
used to seed data/sample_prices.csv so the project still runs end-to-end
in environments without internet access, and so results are reproducible.

Calibration (approx., based on typical long-run behavior of these names):
- Annualized volatility: Reliance ~28%, HDFC Bank ~26%, TCS ~24%, Infosys ~27%, Nifty50 ~16%
- Correlations: single-stock vs Nifty ~0.55-0.70, IT pair (TCS/Infy) ~0.75, Bank vs IT lower ~0.35
- Drift: modest positive equity risk premium, small negative jump risk (fat tails via Student-t innovations)
"""

import numpy as np
import pandas as pd

np.random.seed(42)

TICKERS = ["RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS", "^NSEI"]
N_DAYS = 756  # ~3 trading years

# Annualized vol assumptions
annual_vol = np.array([0.28, 0.26, 0.24, 0.27, 0.16])
daily_vol = annual_vol / np.sqrt(252)

# Small positive daily drift (annualized ~10-14%), Nifty lowest drift/vol as the index
annual_drift = np.array([0.12, 0.13, 0.14, 0.13, 0.11])
daily_drift = annual_drift / 252

# Correlation matrix (symmetric, PSD)
corr = np.array([
    #  RELI  HDFCB  TCS   INFY  NIFTY
    [1.00, 0.35, 0.30, 0.32, 0.68],   # Reliance
    [0.35, 1.00, 0.38, 0.36, 0.66],   # HDFC Bank
    [0.30, 0.38, 1.00, 0.78, 0.60],   # TCS
    [0.32, 0.36, 0.78, 1.00, 0.62],   # Infosys
    [0.68, 0.66, 0.60, 0.62, 1.00],   # Nifty 50
])

cov = np.outer(daily_vol, daily_vol) * corr
L = np.linalg.cholesky(cov)

# Fat-tailed innovations via Student-t (df=5), scaled to unit variance, then correlated
df = 5
t_draws = np.random.standard_t(df, size=(N_DAYS, 5))
t_draws /= np.sqrt(df / (df - 2))  # normalize to unit variance
correlated_shocks = t_draws @ L.T

log_returns = daily_drift + correlated_shocks

# Inject a couple of stress-like clusters (mild "regime" volatility bursts) for realism
# so backtesting has genuine exceedances to find, not just a clean normal series.
stress_windows = [(180, 195), (450, 470)]
for start, end in stress_windows:
    log_returns[start:end] *= 2.3
    log_returns[start:end] -= 0.01  # negative skew during stress

# Build price levels
start_prices = {
    "RELIANCE.NS": 2450.0,
    "HDFCBANK.NS": 1550.0,
    "TCS.NS": 3650.0,
    "INFY.NS": 1480.0,
    "^NSEI": 22000.0,
}

dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=N_DAYS)
prices = pd.DataFrame(index=dates, columns=TICKERS, dtype=float)

for i, tk in enumerate(TICKERS):
    cum_log_ret = np.cumsum(log_returns[:, i])
    prices[tk] = start_prices[tk] * np.exp(cum_log_ret)

prices.index.name = "Date"
prices.round(2).to_csv("/home/claude/market_risk_dashboard/data/sample_prices.csv")
print(prices.tail())
print("\nSaved to data/sample_prices.csv, shape:", prices.shape)
