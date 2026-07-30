"""
data_loader.py
--------------
Fetches historical adjusted-close prices for the portfolio.

Primary path: live data via yfinance.
Fallback path: bundled data/sample_prices.csv (synthetic, offline-safe),
used automatically if yfinance is unreachable or a ticker fails.

This pattern matters in practice: a risk dashboard that hard-fails whenever
a market data vendor has a hiccup is not production-grade.
"""

import os
import pandas as pd

DEFAULT_TICKERS = ["RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS", "^NSEI"]
DISPLAY_NAMES = {
    "RELIANCE.NS": "Reliance",
    "HDFCBANK.NS": "HDFC Bank",
    "TCS.NS": "TCS",
    "INFY.NS": "Infosys",
    "^NSEI": "Nifty 50",
}
SAMPLE_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_prices.csv")


def load_prices(tickers=None, period="3y", use_live=True, verbose=True):
    """
    Returns a DataFrame of adjusted close prices, columns=tickers, indexed by date.

    Parameters
    ----------
    tickers : list[str] or None
        Defaults to DEFAULT_TICKERS.
    period : str
        yfinance period string, e.g. "3y", "1y", "6mo".
    use_live : bool
        If False, skip straight to the sample dataset (useful offline).
    """
    tickers = tickers or DEFAULT_TICKERS

    if use_live:
        try:
            import yfinance as yf
            data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
            if data.empty:
                raise ValueError("yfinance returned no data")
            prices = data["Close"] if "Close" in data.columns.get_level_values(0) else data
            prices = prices.dropna(how="all")
            if prices.isna().all(axis=None):
                raise ValueError("all-NaN price data")
            if verbose:
                print(f"Loaded LIVE data via yfinance: {prices.shape[0]} rows, {prices.shape[1]} tickers")
            return prices.rename(columns=DISPLAY_NAMES)
        except Exception as e:
            if verbose:
                print(f"[data_loader] Live fetch failed ({e}); falling back to bundled sample data.")

    prices = pd.read_csv(SAMPLE_DATA_PATH, index_col="Date", parse_dates=True)
    prices = prices[[t for t in tickers if t in prices.columns]]
    if verbose:
        print(f"Loaded SAMPLE (synthetic, offline) data: {prices.shape[0]} rows, {prices.shape[1]} tickers")
        print("NOTE: this is synthetic data calibrated to realistic vol/correlation, "
              "not real market history. Swap use_live=True with internet access for real analysis.")
    return prices.rename(columns=DISPLAY_NAMES)


if __name__ == "__main__":
    df = load_prices(use_live=True)
    print(df.tail())
