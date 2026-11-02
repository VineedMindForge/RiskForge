import streamlit as st
import pandas as pd
import numpy as np

from src.stress_testing import (
    identify_worst_portfolio_scenarios,
    calculate_historical_stress_attribution,
    plot_asset_stress_attribution,
    plot_stress_pnl_attribution,
    calculate_stress_impact
)

st.set_page_config(page_title="Stress Testing", layout="wide")



st.title("Stress Testing")
st.caption("Evaluate portfolio resilience under historical and hypothetical market stress scenarios.")

if st.session_state.get("portfolio_returns") is None:
    st.warning("Construct a portfolio on the Portfolio Construction page first.")
    st.stop()

portfolio_returns = st.session_state.portfolio_returns.dropna()
portfolio_value = st.session_state.portfolio_value.iloc[-1]

st.sidebar.header("Historical Stress Settings")

period_options = [
    "Full History",
    "COVID-19 Crash (2020)",
    "Inflation and Rate Shock (2022)",
    "Custom Period"
]

selected_period = st.sidebar.selectbox(
    "Historical Period",
    options=period_options
)

if selected_period == "COVID-19 Crash (2020)":
    start_date = pd.Timestamp("2020-02-19")
    end_date = pd.Timestamp("2020-04-30")

elif selected_period == "Inflation and Rate Shock (2022)":
    start_date = pd.Timestamp("2022-01-01")
    end_date = pd.Timestamp("2022-12-31")

elif selected_period == "Custom Period":
    col_start, col_end = st.sidebar.columns(2)

    start_date = col_start.date_input(
        "Start Date",
        value=portfolio_returns.index[0].date(),
        min_value=portfolio_returns.index[0].date(),
        max_value=portfolio_returns.index[-1].date()
    )

    end_date = col_end.date_input(
        "End Date",
        value=portfolio_returns.index[-1].date(),
        min_value=portfolio_returns.index[0].date(),
        max_value=portfolio_returns.index[-1].date()
    )

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    if start_date > end_date:
        st.error("Start date cannot be later than end date.")
        st.stop()

else:
    start_date = portfolio_returns.index[0]
    end_date = portfolio_returns.index[-1]

selected_returns = portfolio_returns.loc[start_date:end_date]

if selected_returns.empty:
    st.error("No portfolio return observations are available for the selected period.")
    st.stop()

if len(selected_returns) < 20:
    st.error("The selected period must contain at least 20 trading observations.")
    st.stop()

st.caption(
    f"Selected period: {selected_returns.index[0]:%Y-%m-%d} to "
    f"{selected_returns.index[-1]:%Y-%m-%d} | "
    f"{len(selected_returns):,} trading observations"
)

st.caption(
    "Historical stress analysis is limited to the investment dates selected "
    "on the Portfolio Construction page. The full downloaded market-data "
    "history is not used for this analysis."
)

stress_results = identify_worst_portfolio_scenarios(
    portfolio_returns=selected_returns,
    portfolio_value=portfolio_value,
    horizons=(1, 5, 20),
    top_n=5
)

def display_stress_metric(label, value, negative=False):
    value_color = "#FF4B4B" if negative else "inherit"

    st.markdown(
        f"""
        <div style="margin-bottom: 4px;">
            <div style="font-size: 14px; color: #A3A8B3;">{label}</div>
            <div style="font-size: 28px; font-weight: 400; color: {value_color};">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()
st.subheader("Historical Portfolio Stress")

st.metric(
    "Current Portfolio Value",
    f"${portfolio_value:,.2f}"
)

col1, col2, col3 = st.columns(3)

for col, horizon in zip([col1, col2, col3], [1, 5, 20]):
    worst_scenario = stress_results[horizon].iloc[0]

    with col:
        display_stress_metric(
            f"Worst {horizon}-Day Return",
            f"{worst_scenario['Stress Return']:.2%}",
            negative=worst_scenario["Stress Return"] < 0
        )

        st.caption(
            f"{worst_scenario['Start Date']:%Y-%m-%d} to "
            f"{worst_scenario['End Date']:%Y-%m-%d}"
        )

        display_stress_metric(
            "Stress P&L",
            f"${worst_scenario['Stress P&L']:,.2f}",
            negative=worst_scenario["Stress P&L"] < 0
        )

        display_stress_metric(
            "Stressed Portfolio Value",
            f"${worst_scenario['Stressed Portfolio Value']:,.2f}"
        )

st.info(
    "Historical stress scenarios apply the identified historical shocks "
    "to the current portfolio value."
)

portfolio_prices = st.session_state.portfolio_prices
portfolio_weights = st.session_state.portfolio_weights_decimal
allocation_drift = st.session_state.allocation_drift

asset_returns = portfolio_prices.pct_change(fill_method=None).dropna()

st.divider()
st.subheader("Asset-Level Stress Attribution")

st.caption(
    "This analysis identifies the portfolio's worst historical return over "
    "a selected holding period. Choose the historical period and stress "
    "horizon using the selectors on the left. Individual Asset Return shows "
    "each asset's compounded performance, while Portfolio Return Contribution "
    "shows its weight-adjusted impact on the overall portfolio return."
)

col_details, col_charts = st.columns([1, 3])

with col_details:
    selected_horizon = st.selectbox(
        "Stress Horizon",
        options=[1, 5, 20],
        format_func=lambda x: f"{x}-Day"
    )

    selected_scenario = st.selectbox(
        "Historical Scenario",
        options=stress_results[selected_horizon].index,
        format_func=lambda x: f"Worst Scenario #{x}"
    )

    scenario = stress_results[selected_horizon].loc[selected_scenario]

    start_date = scenario["Start Date"]
    end_date = scenario["End Date"]

    st.markdown(
        f"""
        <div style="
            background-color: #193047;
            padding: 14px;
            border-radius: 8px;
            margin-top: 20px;
            margin-bottom: 24px;
            min-height: 75px;
        ">
            <div style="font-size: 13px; color: #B8C7D9;">
                Worst-Performing Period
            </div>
            <div style="font-size: 17px; font-weight: 600; color: #4DA8FF; margin-top: 8px;">
                {start_date:%d %b %Y} – {end_date:%d %b %Y}
            </div>
            <div style="font-size: 12px; color: #B8C7D9; margin-top: 5px;">
                Lowest {selected_horizon}-day portfolio return
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; gap: 22px; margin-top: 20px;">
            <div>
                <div style="font-size: 13px; color: #A3A8B3;">Portfolio Stress P&L</div>
                <div style="font-size: 27px; color: #FF4B4B; margin-top: 4px;">
                    ${scenario['Stress P&L']:,.2f}
                </div>
            </div>
            <div>
                <div style="font-size: 13px; color: #A3A8B3;">Stress Return</div>
                <div style="font-size: 27px; color: {'#FF4B4B' if scenario['Stress Return'] < 0 else '#09AB3B'}; margin-top: 4px;">
                    {scenario['Stress Return']:.2%}
                </div>
            </div>
            <div>
                <div style="font-size: 13px; color: #A3A8B3;">Stressed Portfolio Value</div>
                <div style="font-size: 27px; margin-top: 4px;">
                    ${scenario['Stressed Portfolio Value']:,.2f}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_charts:

    attribution = calculate_historical_stress_attribution(
        asset_returns=asset_returns,
        portfolio_returns=portfolio_returns,
        allocation_drift=allocation_drift,
        start_date=start_date,
        end_date=end_date,
        portfolio_value=portfolio_value
    )

    fig = plot_asset_stress_attribution(attribution)
    st.plotly_chart(fig, use_container_width=True)

    fig = plot_stress_pnl_attribution(attribution)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
st.subheader("Custom Stress Scenario")

st.caption(
    "Define hypothetical price movements for individual portfolio assets "
    "to evaluate their combined impact on the portfolio."
)

with st.form("custom_stress_form"):
    scenario_name = st.text_input(
        "Scenario Name",
        value="Custom Scenario"
    )

    st.markdown("**Scenario Portfolio Allocation and Asset Shocks**")

    scenario_weights = {}
    shock_inputs = {}

    for asset in portfolio_weights.index:
        col_asset, col_weight, col_shock = st.columns([1.5, 1.5, 1.5])

        with col_asset:
            st.text_input(
                "Asset",
                value=asset,
                disabled=True,
                key=f"shock_asset_{asset}"
            )

        with col_weight:
            scenario_weights[asset] = st.number_input(
                "Portfolio Weight (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(portfolio_weights[asset] * 100),
                step=1.0,
                key=f"scenario_weight_{asset}"
            )

        with col_shock:
            shock_inputs[asset] = st.number_input(
                "Scenario Shock (%)",
                min_value=-100.0,
                value=0.0,
                step=1.0,
                key=f"shock_input_{asset}"
            )

    total_weight = sum(scenario_weights.values())

    st.caption(f"Total Portfolio Weight: {total_weight:.2f}%")

    submitted = st.form_submit_button(
        "Calculate Stress Impact",
        use_container_width=True
    )

    if submitted:
        if not np.isclose(total_weight, 100.0, atol=0.01):
            st.error(
                f"Portfolio weights must sum to 100%. "
                f"Current total: {total_weight:.2f}%."
            )

        else:
            scenario_weights = pd.Series(scenario_weights) / 100
            scenario_shocks = pd.Series(shock_inputs) / 100

            stress_result = calculate_stress_impact(
                weights=scenario_weights,
                shocks=scenario_shocks,
                portfolio_value=portfolio_value
            )

            st.divider()
            st.subheader(f"{scenario_name} — Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Portfolio Stress Return",
                    f"{stress_result['stress_return']:.2%}",
                    delta_color="inverse"
                )

            with col2:
                st.metric(
                    "Stress P&L",
                    f"${stress_result['stress_pnl']:,.2f}",
                    delta_color="inverse"
                )

            with col3:
                st.metric(
                    "Stressed Portfolio Value",
                    f"${stress_result['stressed_portfolio_value']:,.2f}"
                )

