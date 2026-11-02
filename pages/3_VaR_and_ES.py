
import streamlit as st
import pandas as pd
import numpy as np

from src.value_at_risk import (calculate_var_comparison,  plot_var_es_comparison, plot_return_distribution,
    calculate_var_sensitivity,
    plot_var_sensitivity)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="VaR & Expected Shortfall",
    page_icon="📉",
    layout="wide"
)

st.title("Value at Risk & Expected Shortfall")

st.markdown(
    "Measure portfolio downside risk using Historical, "
    "Parametric and Monte Carlo approaches."
)

st.divider()


# --------------------------------------------------
# CHECK PORTFOLIO DATA
# --------------------------------------------------

required_data = [
    "portfolio_returns",
    "portfolio_prices",
    "portfolio_weights_decimal",
    "portfolio_value"
]

if any(
    st.session_state.get(key) is None
    for key in required_data
):
    st.warning(
        "Construct a portfolio on the Portfolio Construction page first."
    )
    st.stop()


# --------------------------------------------------
# RETRIEVE PORTFOLIO DATA
# --------------------------------------------------

portfolio_returns = st.session_state.portfolio_returns.dropna()

portfolio_prices = st.session_state.portfolio_prices

asset_returns = portfolio_prices.pct_change(
    fill_method=None
).dropna(how="any")

weights = st.session_state.portfolio_weights_decimal

investment_value = st.session_state.portfolio_value.iloc[-1]

valuation_date = st.session_state.portfolio_value.index[-1]


# --------------------------------------------------
# SIDEBAR: RISK SETTINGS
# --------------------------------------------------

st.sidebar.header("Risk Settings")

confidence_level = st.sidebar.selectbox(
    "Confidence Level",
    options=[0.95, 0.99],
    format_func=lambda x: f"{x:.0%}",
    index=0
)

holding_period = st.sidebar.selectbox(
    "Holding Period",
    options=[1, 5, 10, 20],
    format_func=lambda x: f"{x} {'Day' if x == 1 else 'Days'}"
)

lookback_options = [
    "All available",
    250,
    500,
    750
]

lookback = st.sidebar.selectbox(
    "Historical Lookback",
    options=lookback_options
)

window = None if lookback == "All available" else lookback

n_simulations = st.sidebar.selectbox(
    "Monte Carlo Simulations",
    options=[10000, 25000, 50000],
    format_func=lambda x: f"{x:,}"
)


# --------------------------------------------------
# DATA VALIDATION
# --------------------------------------------------

if len(portfolio_returns) < holding_period:
    st.error(
        "Insufficient portfolio return observations "
        "for the selected holding period."
    )
    st.stop()

if len(asset_returns) < 2:
    st.error("Insufficient asset return observations.")
    st.stop()

if window is not None and len(portfolio_returns) < window:
    st.warning(
        f"Only {len(portfolio_returns)} portfolio observations "
        f"are available. The selected lookback of {window} "
        "will be limited to the available history."
    )


# --------------------------------------------------
# CALCULATE VAR AND EXPECTED SHORTFALL
# --------------------------------------------------

try:
    var_comparison = calculate_var_comparison(
        portfolio_returns=portfolio_returns,
        asset_returns=asset_returns,
        weights=weights,
        investment_value=investment_value,
        confidence_level=confidence_level,
        holding_period=holding_period,
        window=window,
        n_simulations=n_simulations
    )

except ValueError as error:
    st.error(f"Unable to calculate portfolio risk: {error}")
    st.stop()


# --------------------------------------------------
# PORTFOLIO RISK SUMMARY
# --------------------------------------------------

st.subheader("Portfolio Risk Summary")

st.caption(
    f"{confidence_level:.0%} confidence level | "
    f"{holding_period}-day holding period | "
    f"Portfolio market value: ${investment_value:,.2f} | "
    f"Valuation date: {valuation_date:%Y-%m-%d}"
)

st.caption(
    "VaR represents the estimated loss threshold. "
    "Expected Shortfall represents the average loss beyond that threshold."
)


# --------------------------------------------------
# RISK METRICS
# --------------------------------------------------

historical, parametric, monte_carlo = st.columns(3)

for column, method in zip(
    [historical, parametric, monte_carlo],
    ["Historical", "Parametric", "Monte Carlo"]
):

    with column:

        result = var_comparison.loc[method]

        st.markdown(f"**{method}**")

        st.metric(
            "Value at Risk",
            f"${result['VaR ($)']:,.2f}",
            delta=None
        )

        st.caption(
            f"{result['VaR (%)']:.2%} of portfolio value"
        )

        st.metric(
            "Expected Shortfall",
            f"${result['ES ($)']:,.2f}",
            delta=None
        )

        st.caption(
            f"{result['ES (%)']:.2%} of portfolio value"
        )


# --------------------------------------------------
# MODEL ASSUMPTIONS
# --------------------------------------------------

st.caption(
    "Monte Carlo assumes daily portfolio rebalancing during "
    "the simulated holding period. Its results do not currently "
    "reflect the rebalancing frequency selected on the Portfolio "
    "Construction page."
)


st.divider()


# --------------------------------------------------
# VAR AND ES COMPARISON CHART
# --------------------------------------------------

st.subheader("VaR and Expected Shortfall Comparison")

st.caption(
    "Compares portfolio Value at Risk (VaR) and Expected Shortfall (ES) "
    "across Historical, Parametric and Monte Carlo estimation methods. "
    "VaR represents the loss threshold at the selected confidence level, "
    "while ES measures the average loss beyond that threshold."
)


fig_comparison = plot_var_es_comparison(var_comparison)

st.plotly_chart(
    fig_comparison,
    use_container_width=True
)



# --------------------------------------------------
# PORTFOLIO RETURN DISTRIBUTION
# --------------------------------------------------

st.divider()

st.subheader("Portfolio Return Distribution")
st.caption(
    "Displays the portfolio return distribution over the selected holding period, "
    "with the VaR threshold and Expected Shortfall highlighted for the selected model. "
    "Switch between Historical, Parametric and Monte Carlo to compare their estimated "
    "downside risk at the selected confidence level."
)


selected_model = st.selectbox(
    "Select model to highlight",
    options=["Historical", "Parametric", "Monte Carlo"]
)

fig_distribution = plot_return_distribution(
    portfolio_returns=portfolio_returns,
    var_comparison=var_comparison,
    selected_model=selected_model,
    holding_period=holding_period
)

st.plotly_chart(
    fig_distribution,
    use_container_width=True
)



# --------------------------------------------------
# VAR SENSITIVITY ANALYSIS
# --------------------------------------------------

st.divider()

st.subheader("VaR Sensitivity Analysis")


st.caption(
    "Compares how portfolio VaR changes across different confidence levels "
    "using Historical, Parametric and Monte Carlo methods. "
    "We generally expect the slope to become steeper beyond 95%, "
    "as higher confidence levels capture increasingly extreme potential losses."
)

sensitivity_data = calculate_var_sensitivity(
    portfolio_returns=portfolio_returns,
    asset_returns=asset_returns,
    weights=weights,
    investment_value=investment_value,
    holding_period=holding_period,
    window=window,
    n_simulations=n_simulations
)

fig_sensitivity = plot_var_sensitivity(sensitivity_data)

st.plotly_chart(
    fig_sensitivity,
    use_container_width=True
)
