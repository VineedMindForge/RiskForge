import streamlit as st
import pandas as pd
from src.value_at_risk import historical_var_es
from src.drawdown import calculate_drawdown, calculate_drawdown_metrics, plot_drawdown



st.set_page_config(
    page_title="Executive Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Executive Dashboard")
st.caption(
    "A consolidated view of portfolio performance, market risk exposure "
    "and downside risk indicators across the selected investment period."
)


# --------------------------------------------------
# PORTFOLIO DATA
# --------------------------------------------------

portfolio_value = st.session_state.get("portfolio_value")
portfolio_metrics = st.session_state.get("portfolio_metrics")
portfolio_returns = st.session_state.get("portfolio_returns")


if portfolio_value is None or portfolio_metrics is None or portfolio_returns is None:
    st.warning("Construct a portfolio on the Portfolio Construction page first.")
    st.stop()


# --------------------------------------------------
# PORTFOLIO KPIs
# --------------------------------------------------

current_value = float(portfolio_value.iloc[-1])
previous_value = float(portfolio_value.iloc[-2]) if len(portfolio_value) > 1 else current_value

daily_pnl = current_value - previous_value
daily_return = daily_pnl / previous_value if previous_value else 0

total_return = portfolio_metrics["Total Return"]
cagr = portfolio_metrics["CAGR"]
annualized_volatility = portfolio_metrics["Annualized Volatility"]
sharpe_ratio = portfolio_metrics["Sharpe Ratio"]


# --------------------------------------------------
# KPI DISPLAY
# --------------------------------------------------

st.subheader("Portfolio Overview")
st.caption(
    "A summary of portfolio performance, value and risk-adjusted returns "
    "over the selected investment period."
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Portfolio Value",
    f"${current_value:,.2f}",
    f"{daily_return:.2%}"
)

col2.metric(
    "Daily P&L",
    f"${daily_pnl:,.2f}",
    f"{daily_return:.2%}"
)

col3.metric(
    "Total Return",
    f"{total_return:.2%}",
    f"CAGR: {cagr:.2%}"
)

col4.metric(
    "Annualized Volatility",
    f"{annualized_volatility:.2%}"
)


col1, col2 = st.columns(2)

col1.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
col2.metric("Initial Investment", f"${st.session_state.initial_investment:,.2f}")


# --------------------------------------------------
# PORTFOLIO PERFORMANCE
# --------------------------------------------------

st.divider()
st.subheader("Portfolio Performance")
st.caption(
    "Tracks the evolution of portfolio value over the investment period, "
    "highlighting overall growth and changes in portfolio performance."
)

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=portfolio_value.index,
    y=portfolio_value.values,
    mode="lines",
    name="Portfolio Value",
    line=dict(width=2)
))

fig.update_layout(
    template="plotly_white",
    height=400,
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title="Portfolio Value ($)",
    yaxis_tickprefix="$",
    yaxis_tickformat=",.0f",
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# RISK CALCULATIONS
# --------------------------------------------------

# Historical VaR
var_result = historical_var_es(
    portfolio_returns=portfolio_returns,
    investment_value=current_value,
    confidence_level=0.95,
    holding_period=1,
    window=252
)

# Drawdown calculations
drawdown_results = calculate_drawdown(portfolio_value)

drawdown = drawdown_results["drawdown"]
running_peak = drawdown_results["running_peak"]

drawdown_metrics = calculate_drawdown_metrics(portfolio_value, drawdown)

# Latest daily performance
latest_daily_return = portfolio_returns.iloc[-1]
latest_daily_pnl = daily_pnl

# --------------------------------------------------
# RISK SNAPSHOT DISPLAY
# --------------------------------------------------

st.divider()
st.subheader("Risk Snapshot")
st.caption(
    "A snapshot of portfolio downside risk, latest daily performance "
    "and drawdown characteristics."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Historical VaR (95%)",
        f"${var_result['var_amount']:,.2f}"
    )
    st.caption("1-day | 252-day lookback")

with col2:
    st.metric(
        "Latest Daily Return",
        f"{latest_daily_return:.2%}"
    )
    st.caption(f"Daily P&L: ${latest_daily_pnl:,.2f}")

with col3:
    st.metric(
        "Current Drawdown",
        f"{drawdown_metrics['current_drawdown']:.2%}"
    )

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Maximum Drawdown",
        f"{drawdown_metrics['max_drawdown']:.2%}"
    )

with col2:
    st.metric(
        "Recovery Status",
        drawdown_metrics["recovery_status"]
    )

# --------------------------------------------------
# DRAWDOWN ANALYSIS
# --------------------------------------------------

st.divider()
st.subheader("Portfolio Drawdown")

drawdown_fig = plot_drawdown(drawdown)

st.plotly_chart(drawdown_fig, use_container_width=True)

# --------------------------------------------------
# DAILY LOSS VS HISTORICAL VAR
# --------------------------------------------------

st.divider()
st.subheader("Daily Loss vs Historical VaR")
st.caption(
    "Compares the latest daily portfolio loss against the Historical "
    "Simulation VaR threshold to assess whether the observed loss "
    "exceeded the estimated risk level."
)

daily_loss = max(0, -latest_daily_pnl)
var_utilization = daily_loss / var_result["var_amount"] if var_result["var_amount"] > 0 else 0

col1, col2 = st.columns(2)

col1.metric("Latest Daily Loss", f"${daily_loss:,.2f}")
col2.metric("Historical VaR Threshold", f"${var_result['var_amount']:,.2f}")

st.metric(
    "Daily Loss / VaR",
    f"{var_utilization:.1%}"
)

if latest_daily_pnl >= 0:
    st.success("Portfolio recorded a gain on the latest trading day.")
elif var_utilization >= 1:
    st.error("Latest daily loss exceeded the Historical VaR threshold.")
else:
    st.info("Latest daily loss remained within the Historical VaR threshold.")