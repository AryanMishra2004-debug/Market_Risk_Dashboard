"""
risk_metrics.py
---------------
Core VaR / Expected Shortfall engine for a multi-asset portfolio.

Conventions:
- Returns are log returns: r_t = ln(P_t / P_{t-1})
- VaR and ES are reported as POSITIVE numbers representing a loss
  (i.e. "VaR_95 = 2.3%" means a 2.3% loss is the 95th-percentile bad day).
- Portfolio return = weighted sum of asset log returns (standard small-time-step approximation).
"""

import numpy as np
import pandas as pd
from scipy import stats


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")


def portfolio_returns(returns: pd.DataFrame, weights: dict) -> pd.Series:
    """weights: dict of {column_name: weight}, need not be normalized (will be normalized here)."""
    w = pd.Series(weights)
    w = w / w.sum()
    cols = [c for c in w.index if c in returns.columns]
    return (returns[cols] * w[cols]).sum(axis=1)


def annualized_vol(returns: pd.Series, periods=252) -> float:
    return returns.std() * np.sqrt(periods)


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


# ---------------------------------------------------------------------------
# VaR methods
# ---------------------------------------------------------------------------

def historical_var(returns: pd.Series, confidence=0.95, horizon_days=1):
    """
    Empirical (historical simulation) VaR: the (1-confidence) percentile
    of the empirical return distribution, scaled to the horizon by sqrt(t).
    """
    q = np.percentile(returns.dropna(), (1 - confidence) * 100)
    var_1day = -q
    return var_1day * np.sqrt(horizon_days)


def historical_es(returns: pd.Series, confidence=0.95, horizon_days=1):
    """Expected Shortfall = average loss in the tail beyond the VaR threshold."""
    q = np.percentile(returns.dropna(), (1 - confidence) * 100)
    tail = returns[returns <= q]
    es_1day = -tail.mean() if len(tail) > 0 else -q
    return es_1day * np.sqrt(horizon_days)


def parametric_var(returns: pd.Series, confidence=0.95, horizon_days=1):
    """
    Variance-covariance (delta-normal) VaR assuming normally distributed returns.
    VaR = -(mu + z * sigma), z is the negative-tail z-score.
    """
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - confidence)  # negative value
    var_1day = -(mu + z * sigma)
    return var_1day * np.sqrt(horizon_days)


def parametric_es(returns: pd.Series, confidence=0.95, horizon_days=1):
    """Closed-form ES under normality: ES = -(mu - sigma * phi(z)/(1-confidence))."""
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - confidence)
    es_1day = -(mu - sigma * stats.norm.pdf(z) / (1 - confidence))
    return es_1day * np.sqrt(horizon_days)


def monte_carlo_var(returns: pd.DataFrame, weights: dict, confidence=0.95,
                     horizon_days=1, n_sims=50_000, dist="t", t_df=5, seed=42):
    """
    Simulates correlated portfolio returns via the historical covariance matrix
    (Cholesky decomposition) and computes VaR/ES from the simulated distribution.

    dist="t" uses Student-t innovations (fatter tails, more realistic for equities)
    dist="normal" uses Gaussian innovations (matches parametric method as a sanity check).
    """
    rng = np.random.default_rng(seed)
    w = pd.Series(weights)
    w = w / w.sum()
    cols = [c for c in w.index if c in returns.columns]
    r = returns[cols].dropna()
    mu = r.mean().values
    cov = r.cov().values
    L = np.linalg.cholesky(cov)

    if dist == "t":
        raw = rng.standard_t(t_df, size=(n_sims, len(cols)))
        raw /= np.sqrt(t_df / (t_df - 2))  # normalize to unit variance
    else:
        raw = rng.standard_normal(size=(n_sims, len(cols)))

    sim_asset_returns = mu + raw @ L.T
    sim_portfolio_returns = sim_asset_returns @ w[cols].values

    q = np.percentile(sim_portfolio_returns, (1 - confidence) * 100)
    var_1day = -q
    tail = sim_portfolio_returns[sim_portfolio_returns <= q]
    es_1day = -tail.mean()

    return {
        "VaR": var_1day * np.sqrt(horizon_days),
        "ES": es_1day * np.sqrt(horizon_days),
        "simulated_returns": sim_portfolio_returns,
    }


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

def stress_test(returns: pd.DataFrame, weights: dict, scenarios: dict):
    """
    scenarios: dict of {scenario_name: {asset_name: shock_return}}
    shock_return is a log return, e.g. -0.10 for a 10% single-day drop.
    Any asset not specified in a scenario is assumed unchanged (shock=0).
    Returns a DataFrame of portfolio P&L (%) under each scenario.
    """
    w = pd.Series(weights)
    w = w / w.sum()
    cols = [c for c in w.index if c in returns.columns]

    results = {}
    for name, shocks in scenarios.items():
        shock_vec = pd.Series({c: shocks.get(c, 0.0) for c in cols})
        pnl = (shock_vec * w[cols]).sum()
        results[name] = pnl
    return pd.Series(results, name="portfolio_pnl_pct") * 100


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def rolling_var_backtest(returns: pd.Series, confidence=0.95, window=250, method="historical"):
    """
    Rolling-window backtest: for each day t (after the first `window` days),
    compute VaR using only data up to t-1, then compare against the actual
    realized return on day t. Returns a DataFrame with columns:
    ['actual_return', 'predicted_var', 'exceedance'].
    """
    dates = returns.index[window:]
    predicted = []
    actual = []

    for i in range(window, len(returns)):
        hist_window = returns.iloc[i - window:i]
        if method == "historical":
            v = historical_var(hist_window, confidence)
        elif method == "parametric":
            v = parametric_var(hist_window, confidence)
        else:
            raise ValueError("method must be 'historical' or 'parametric'")
        predicted.append(v)
        actual.append(returns.iloc[i])

    df = pd.DataFrame({
        "actual_return": actual,
        "predicted_var": predicted,
    }, index=dates)
    # Exceedance: actual loss (negative return) worse than predicted VaR threshold
    df["exceedance"] = df["actual_return"] < -df["predicted_var"]
    return df


def kupiec_pof_test(exceedances: pd.Series, confidence=0.95):
    """
    Kupiec Proportion-of-Failures test: checks whether the observed exceedance
    rate is statistically consistent with the expected failure rate (1-confidence).
    Returns (LR_statistic, p_value, expected_rate, observed_rate).
    Under H0 the LR statistic is chi-square distributed with 1 degree of freedom.
    """
    n = len(exceedances)
    x = exceedances.sum()  # number of exceedances
    p = 1 - confidence

    if x == 0 or x == n:
        # Avoid log(0); use a tiny epsilon
        p_hat = max(min(x / n, 1 - 1e-6), 1e-6)
    else:
        p_hat = x / n

    ll_null = (n - x) * np.log(1 - p) + x * np.log(p)
    ll_alt = (n - x) * np.log(1 - p_hat) + x * np.log(p_hat)
    lr_stat = -2 * (ll_null - ll_alt)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    return {
        "n_obs": n,
        "n_exceedances": int(x),
        "expected_rate": p,
        "observed_rate": x / n,
        "LR_statistic": lr_stat,
        "p_value": p_value,
        "reject_null_at_5pct": p_value < 0.05,  # True => model likely miscalibrated
    }
