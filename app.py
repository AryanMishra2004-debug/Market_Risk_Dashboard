"""
Market Risk Analytics Dashboard — VaR & Expected Shortfall
Run with: streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from data_loader import load_prices, DEFAULT_TICKERS
from risk_metrics import (
    log_returns, portfolio_returns, annualized_vol, correlation_matrix,
    historical_var, historical_es, parametric_var, parametric_es,
    monte_carlo_var, stress_test, rolling_var_backtest, kupiec_pof_test
)

st.set_page_config(page_title="Market Risk Analytics Dashboard", layout="wide")

st.title("📊 Market Risk Analytics Dashboard")
st.caption("Value-at-Risk & Expected Shortfall — Reliance · HDFC Bank · TCS · Infosys · Nifty 50")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("Configuration")

use_live = st.sidebar.toggle("Attempt live data (yfinance)", value=True,
                              help="Falls back automatically to bundled sample data if unavailable.")
period = st.sidebar.selectbox("History window", ["1y", "2y", "3y", "5y"], index=2)

st.sidebar.subheader("Portfolio Weights")
default_w = {"Reliance": 25, "HDFC Bank": 25, "TCS": 20, "Infosys": 15, "Nifty 50": 15}
weights_pct = {}
for name, default in default_w.items():
    weights_pct[name] = st.sidebar.slider(name, 0, 100, default, 5)

total_w = sum(weights_pct.values())
if total_w == 0:
    st.sidebar.error("At least one weight must be non-zero.")
    st.stop()
weights = {k: v / total_w for k, v in weights_pct.items()}
st.sidebar.caption(f"Normalized to 100% (raw sum: {total_w}%)")

confidence = st.sidebar.select_slider("Confidence level", options=[0.90, 0.95, 0.975, 0.99], value=0.95)
horizon = st.sidebar.number_input("Horizon (trading days)", min_value=1, max_value=20, value=1)
mc_sims = st.sidebar.select_slider("Monte Carlo simulations", options=[5_000, 10_000, 25_000, 50_000, 100_000], value=50_000)
mc_dist = st.sidebar.radio("Monte Carlo innovation distribution", ["t", "normal"], index=0,
                            help="Student-t has fatter tails, more realistic for equity returns.")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading price data...")
def get_data(period, use_live):
    prices = load_prices(period=period, use_live=use_live, verbose=False)
    return prices

prices = get_data(period, use_live)
returns = log_returns(prices)
port_ret = portfolio_returns(returns, weights)

data_source_note = ("⚠️ Using bundled **synthetic sample data** (offline fallback) — "
                     "not real market history." if prices.shape[0] < 5 or not use_live
                     else "Data source: attempted live yfinance fetch (falls back silently to sample data if unreachable).")
st.info(data_source_note)

# ---------------------------------------------------------------------------
# Top-line metrics
# ---------------------------------------------------------------------------
h_var = historical_var(port_ret, confidence, horizon)
h_es = historical_es(port_ret, confidence, horizon)
p_var = parametric_var(port_ret, confidence, horizon)
p_es = parametric_es(port_ret, confidence, horizon)
mc = monte_carlo_var(returns, weights, confidence=confidence, horizon_days=horizon,
                      n_sims=mc_sims, dist=mc_dist)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(f"Historical VaR ({int(confidence*100)}%)", f"{h_var*100:.2f}%")
    st.metric("Historical ES", f"{h_es*100:.2f}%")
with col2:
    st.metric(f"Parametric VaR ({int(confidence*100)}%)", f"{p_var*100:.2f}%")
    st.metric("Parametric ES", f"{p_es*100:.2f}%")
with col3:
    st.metric(f"Monte Carlo VaR ({int(confidence*100)}%)", f"{mc['VaR']*100:.2f}%")
    st.metric("Monte Carlo ES", f"{mc['ES']*100:.2f}%")

st.caption(f"Portfolio annualized volatility: **{annualized_vol(port_ret)*100:.2f}%** &nbsp;|&nbsp; "
           f"Horizon: {horizon} trading day(s) &nbsp;|&nbsp; Confidence: {int(confidence*100)}%")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Prices & Correlation", "🎲 Monte Carlo Distribution",
                                    "⚡ Stress Testing", "✅ Backtesting"])

with tab1:
    c1, c2 = st.columns([3, 2])
    with c1:
        norm_prices = prices / prices.iloc[0] * 100
        fig = px.line(norm_prices, title="Normalized Price History (base=100)")
        fig.update_layout(yaxis_title="Indexed level", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        corr = correlation_matrix(returns)
        fig2 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                          title="Correlation Matrix (daily log returns)")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=mc["simulated_returns"] * 100, nbinsx=100,
                                  marker_color="#3B6EA5", name="Simulated returns"))
    fig3.add_vline(x=-mc["VaR"] * 100, line_dash="dash", line_color="crimson",
                    annotation_text=f"VaR {confidence*100:.0f}% = {mc['VaR']*100:.2f}%")
    fig3.add_vline(x=-mc["ES"] * 100, line_dash="dash", line_color="darkorange",
                    annotation_text=f"ES {confidence*100:.0f}% = {mc['ES']*100:.2f}%")
    fig3.update_layout(title=f"Monte Carlo Simulated Portfolio Returns ({mc_sims:,} paths, {mc_dist}-distributed)",
                        xaxis_title="Simulated 1-day portfolio return (%)", yaxis_title="Frequency")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Fat-tailed (Student-t) innovations produce more realistic extreme-loss scenarios "
               "than a pure Gaussian assumption.")

with tab3:
    st.subheader("Hypothetical Shock Scenarios")
    st.caption("Deterministic 'what-if' shocks applied directly to asset returns — no probability attached.")

    default_scenarios = {
        "2008-style broad crash": {"Reliance": -15, "TCS": -15, "Infosys": -15, "Nifty 50": -15, "HDFC Bank": -8},
        "IT sector selloff": {"TCS": -12, "Infosys": -12},
        "COVID-style flash crash (-10% broad)": {c: -10 for c in returns.columns},
        "Banking-specific crisis": {"HDFC Bank": -20},
    }
    scenarios = {name: {k: v / 100 for k, v in shocks.items()} for name, shocks in default_scenarios.items()}
    stress_results = stress_test(returns, weights, scenarios)

    fig4 = px.bar(stress_results, orientation="h",
                   color=stress_results.values, color_continuous_scale="RdYlGn",
                   labels={"value": "Portfolio P&L (%)", "index": ""},
                   title="Stress Test Impact on Portfolio")
    fig4.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)
    st.dataframe(stress_results.to_frame("Portfolio P&L (%)").style.format("{:.2f}"))

with tab4:
    st.subheader("Rolling VaR Backtest")
    window = st.slider("Rolling window (days)", 100, 500, 250, 50)
    method = st.radio("VaR method to backtest", ["historical", "parametric"], horizontal=True)

    with st.spinner("Running rolling backtest..."):
        bt = rolling_var_backtest(port_ret, confidence=confidence, window=window, method=method)

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=bt.index, y=bt["actual_return"] * 100, name="Actual daily return",
                                line=dict(color="steelblue", width=1)))
    fig5.add_trace(go.Scatter(x=bt.index, y=-bt["predicted_var"] * 100, name=f"Predicted -VaR ({int(confidence*100)}%)",
                                line=dict(color="crimson", width=1.2)))
    exceed = bt[bt.exceedance]
    fig5.add_trace(go.Scatter(x=exceed.index, y=exceed["actual_return"] * 100, mode="markers",
                                marker=dict(color="black", size=7), name="Exceedance"))
    fig5.update_layout(title=f"Rolling {int(confidence*100)}% VaR Backtest ({method.title()} method, {window}-day window)",
                        yaxis_title="Return (%)")
    st.plotly_chart(fig5, use_container_width=True)

    kt = kupiec_pof_test(bt["exceedance"], confidence=confidence)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Observations", kt["n_obs"])
    k2.metric("Exceedances", kt["n_exceedances"])
    k3.metric("Expected rate", f"{kt['expected_rate']*100:.1f}%")
    k4.metric("Observed rate", f"{kt['observed_rate']*100:.1f}%")

    if kt["reject_null_at_5pct"]:
        st.error(f"⚠️ Kupiec test: model likely **miscalibrated** (p-value = {kt['p_value']:.4f}). "
                 f"Observed exceedance rate deviates significantly from the expected rate.")
    else:
        st.success(f"✅ Kupiec test: model is **well-calibrated** (p-value = {kt['p_value']:.4f}). "
                   f"Observed exceedance rate is statistically consistent with the {int(confidence*100)}% confidence level.")

st.divider()
st.caption("Built with Python, pandas, NumPy, SciPy, Plotly, and Streamlit. "
           "Historical, Parametric (variance-covariance), and Monte Carlo (Student-t) VaR "
           "methodologies, cross-validated via Kupiec backtesting.")
