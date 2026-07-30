# Market_Risk_Dashboard
# Market Risk Analytics Dashboard — VaR & Expected Shortfall

A quantitative risk analytics tool for a multi-asset equity portfolio (Reliance, HDFC Bank,
TCS, Infosys, Nifty 50), answering the core market risk question:

> **How much can this portfolio potentially lose over the next trading day, at 95%/99% confidence?**

## What's in this project

| Component | What it does |
|---|---|
| `src/data_loader.py` | Loads historical prices via `yfinance`, with automatic offline fallback to bundled synthetic data |
| `src/risk_metrics.py` | Core engine: Historical, Parametric, and Monte Carlo VaR; Expected Shortfall; correlation; stress testing; rolling backtesting; Kupiec test |
| `notebooks/methodology.ipynb` | Full walkthrough with narrative, plots, and interpretation — start here to understand the methodology |
| `app.py` | Interactive Streamlit dashboard — adjustable weights, confidence levels, VaR method comparison, live charts |
| `src/generate_sample_data.py` | Generates the offline fallback dataset (synthetic, calibrated to realistic vol/correlation) |

## Methods implemented

- **Historical VaR** — empirical percentile of past returns, no distributional assumption
- **Parametric (Variance-Covariance) VaR** — closed-form under normality assumption
- **Monte Carlo VaR** — 50,000-path simulation using Student-t (fat-tailed) innovations via Cholesky-decomposed covariance
- **Expected Shortfall (CVaR)** — average loss beyond the VaR threshold, for every method above
- **Stress testing** — deterministic shock scenarios (e.g. 2008-style crash, sector-specific selloff)
- **Backtesting** — rolling out-of-sample VaR estimation + Kupiec Proportion-of-Failures test to validate model calibration

## Setup

```bash
pip install -r requirements.txt
```

## Run the notebook

```bash
jupyter notebook notebooks/methodology.ipynb
```

## Run the dashboard

```bash
streamlit run app.py
```

Notes on data
The primary data path uses `yfinance` for real historical NSE prices
(`RELIANCE.NS`, `HDFCBANK.NS`, `TCS.NS`, `INFY.NS`, `^NSEI`). If `yfinance` is unreachable
(no internet, rate limit, ticker outage), the loader automatically falls back to
`data/sample\_prices.csv` — a synthetic dataset calibrated to realistic volatility and
correlation for these five assets, so the project always runs end-to-end. This fallback
behavior is itself intentional: a risk pipeline that hard-fails on a data vendor hiccup
isn't production-realistic.

Possible extensions
Add historical stress scenarios calibrated to actual past crisis windows (2008, 2020) instead of hypothetical shocks
Extend to a multi-day VaR term structure rather than sqrt(time) scaling
Add options/derivatives with non-linear payoffs (Monte Carlo already supports this extension; Parametric/Historical do not)
Component VaR / marginal VaR to show each asset's contribution to total portfolio risk



correlation for these five assets, so the project always runs end-to-end. This fallback
behavior is itself intentional: a risk pipeline that hard-fails on a data vendor hiccup
isn't production-realistic.
