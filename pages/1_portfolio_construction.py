import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.market_data import download_data

from src.portfolio_construction import (
    calculate_portfolio_returns,
    calculate_portfolio_value,
    calculate_portfolio_metrics,
    plot_portfolio_vs_assets,
    plot_allocation_over_time,
    plot_current_allocation
)

st.write("Available assets initialized:", "available_assets" in st.session_state)
# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Portfolio Construction",
    page_icon="📊",
    layout="wide"
)

st.title("Portfolio Construction")
st.markdown(
    "Construct and analyze an investment portfolio using different "
    "asset allocation and rebalancing strategies."
)

st.divider()


# --------------------------------------------------
# MARKET DATA
# --------------------------------------------------

@st.cache_data
def get_market_data(tickers):
    return download_data(tickers, period="10y")


# --------------------------------------------------
# VISUALIZATION SETTINGS
# --------------------------------------------------

color_palette = px.colors.qualitative.Dark24
portfolio_color = "#FF3030"


def apply_asset_colors(fig, asset_colors, portfolio_color=None):

    for trace in fig.data:

        if trace.type == "pie":
            labels = list(trace.labels) if trace.labels is not None else []
            trace.marker.colors = [
                asset_colors.get(str(label), "#808080")
                for label in labels
            ]

        else:
            trace_name = str(trace.name)

            if trace_name == "Portfolio" and portfolio_color:
                color = portfolio_color
            else:
                color = asset_colors.get(trace_name)

            if color:
                trace.line.color = color

                if trace.marker is not None:
                    trace.marker.color = color

                if trace.fill and trace.fill != "none":
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)
                    trace.fillcolor = f"rgba({r}, {g}, {b}, 0.45)"

    return fig


# --------------------------------------------------
# SHARED STATE CALLBACKS
# --------------------------------------------------

def save_selected_assets():
    st.session_state.selected_assets = (
        st.session_state.portfolio_selected_assets_widget
    )


def save_portfolio_weights():

    saved_weights = dict(st.session_state.portfolio_weights)

    for asset in st.session_state.selected_assets:
        saved_weights[asset] = st.session_state[f"weight_{asset}"]

    st.session_state.portfolio_weights = saved_weights


def save_investment_dates():
    st.session_state.start_date = st.session_state.portfolio_start_date_widget
    st.session_state.end_date = st.session_state.portfolio_end_date_widget


def save_initial_investment():
    st.session_state.initial_investment = (
        st.session_state.portfolio_initial_investment_widget
    )


def save_rebalance_frequency():
    st.session_state.rebalance_frequency = (
        st.session_state.portfolio_rebalance_widget
    )


# --------------------------------------------------
# SIDEBAR: PORTFOLIO SETTINGS
# --------------------------------------------------

if "available_assets" not in st.session_state:
    stock_universe = pd.read_csv("data/stock_universe.csv")
    st.session_state.available_assets = stock_universe["ticker"].dropna().astype(str).tolist()

available_assets = st.session_state.available_assets

asset_colors = {
    asset: color_palette[i % len(color_palette)]
    for i, asset in enumerate(available_assets)
}

st.sidebar.header("Portfolio Settings")


# --------------------------------------------------
# ASSET SELECTION
# --------------------------------------------------

def save_selected_assets():
    st.session_state.selected_assets = (
        st.session_state.portfolio_selected_assets_widget
    )


if "portfolio_selected_assets_widget" not in st.session_state:
    st.session_state.portfolio_selected_assets_widget = (
        st.session_state.selected_assets
    )


selected_assets = st.sidebar.multiselect(
    "Select stocks",
    options=available_assets,
    max_selections=4,
    placeholder="Search and select up to 4 stocks",
    key="portfolio_selected_assets_widget",
    on_change=save_selected_assets
)

if not selected_assets:
    st.info("Select up to 4 stocks to construct your portfolio.")
    st.stop()


# --------------------------------------------------
# PORTFOLIO WEIGHTS
# --------------------------------------------------

st.sidebar.subheader("Portfolio Weights")

# Reset weights only when the selected assets change.
if (
    "portfolio_weights_assets" not in st.session_state
    or st.session_state.portfolio_weights_assets != list(selected_assets)
):

    equal_weight = round(100.0 / len(selected_assets), 2)

    new_weights = {
        asset: equal_weight
        for asset in selected_assets
    }

    # Adjust the final weight to ensure the total is exactly 100%.
    last_asset = selected_assets[-1]
    new_weights[last_asset] = round(
        100.0 - sum(
            weight for asset, weight in new_weights.items()
            if asset != last_asset
        ),
        2
    )

    st.session_state.portfolio_weights = new_weights
    st.session_state.portfolio_weights_assets = list(selected_assets)

    # Synchronize the individual number-input widgets.
    for asset, weight in new_weights.items():
        st.session_state[f"weight_{asset}"] = weight


def save_portfolio_weights():

    st.session_state.portfolio_weights = {
        asset: st.session_state[f"weight_{asset}"]
        for asset in st.session_state.selected_assets
    }


portfolio_weights = {}

for asset in selected_assets:

    widget_key = f"weight_{asset}"

    # Restore widget value from persistent portfolio weights.
    if widget_key not in st.session_state:
        st.session_state[widget_key] = (
            st.session_state.portfolio_weights.get(asset, 0.0)
        )

    portfolio_weights[asset] = st.sidebar.number_input(
        f"{asset} Weight (%)",
        min_value=0.0,
        max_value=100.0,
        step=1.0,
        key=widget_key,
        on_change=save_portfolio_weights
    )

st.session_state.portfolio_weights = portfolio_weights

# --------------------------------------------------
# INVESTMENT SETTINGS
# --------------------------------------------------

st.sidebar.subheader("Investment Settings")

if "portfolio_start_date_widget" not in st.session_state:
    st.session_state.portfolio_start_date_widget = st.session_state.start_date

if "portfolio_end_date_widget" not in st.session_state:
    st.session_state.portfolio_end_date_widget = st.session_state.end_date

start_date = st.sidebar.date_input(
    "Investment Start Date",
    key="portfolio_start_date_widget",
    on_change=save_investment_dates
)

end_date = st.sidebar.date_input(
    "Investment End Date",
    key="portfolio_end_date_widget",
    on_change=save_investment_dates
)

if start_date >= end_date:
    st.sidebar.error("End date must be after start date.")
    st.stop()


# --------------------------------------------------
# INITIAL INVESTMENT
# --------------------------------------------------

if "portfolio_initial_investment_widget" not in st.session_state:
    st.session_state.portfolio_initial_investment_widget = (
        st.session_state.initial_investment
    )

initial_investment = st.sidebar.number_input(
    "Initial Investment ($)",
    min_value=1000.0,
    step=1000.0,
    key="portfolio_initial_investment_widget",
    on_change=save_initial_investment
)


# --------------------------------------------------
# REBALANCING SETTINGS
# --------------------------------------------------

st.sidebar.subheader("Rebalancing")

if "portfolio_rebalance_widget" not in st.session_state:
    st.session_state.portfolio_rebalance_widget = (
        st.session_state.rebalance_frequency
    )

rebalance_frequency = st.sidebar.selectbox(
    "Rebalancing Frequency",
    options=["Buy and Hold", "Monthly", "Daily"],
    key="portfolio_rebalance_widget",
    on_change=save_rebalance_frequency,
    help="Choose how frequently the portfolio returns to its target asset weights."
)


# --------------------------------------------------
# DOWNLOAD MARKET DATA
# --------------------------------------------------

st.subheader("Market Data")

st.caption(
    "Historical price data used to construct the portfolio "
    "and calculate its performance and risk metrics."
)

with st.spinner("Downloading up to 10 years of market data..."):

    market_prices, quality_report, has_warnings = get_market_data(
        tuple(selected_assets)
    )

# Keep the full history available to the other pages.
st.session_state.market_prices = market_prices
st.session_state.market_quality_report = quality_report
st.session_state.market_data_warning = has_warnings
st.session_state.market_data_assets = list(selected_assets)

# Keep these names temporarily because the existing analytics page uses them.
st.session_state.prices = market_prices
st.session_state.quality_report = quality_report
st.session_state.has_warnings = has_warnings

# --------------------------------------------------
# DATA QUALITY REPORT
# --------------------------------------------------

with st.expander("Data Quality Report", expanded=False):

    display_quality_report = quality_report.drop(
        columns=["Latest Valid Price", "Latest Valid Price Date"],
        errors="ignore"
    ).copy()

    for date_column in ["First Date", "Last Date"]:
        display_quality_report[date_column] = (
            pd.to_datetime(display_quality_report[date_column])
            .dt.strftime("%Y-%m-%d")
        )

    display_quality_report["Data Warning"] = (
        display_quality_report["Data Warning"]
        .map({True: "Warning", False: "No issues"})
    )

    if has_warnings:
        st.warning(
            "Data quality warnings detected within the available history. "
            "Dates before an asset's first available price are not treated as missing data."
        )
        table_color = "#4A3514"
    else:
        st.success("No data-quality issues detected within the available history.")
        table_color = "#123524"

    def highlight_quality(row):
        if row["Data Warning"] == "Warning":
            return [
                f"background-color: {table_color}; color: #F8FAFC"
            ] * len(row)

        return [""] * len(row)

    styled_quality_report = display_quality_report.style.apply(
        highlight_quality,
        axis=1
    )

    st.dataframe(
        styled_quality_report,
        use_container_width=True
    )


# --------------------------------------------------
# SELECT THE INVESTMENT PERIOD
# --------------------------------------------------

# Keep only the dates selected by the user, then retain dates where every
# selected asset has a valid price. This creates a common history for the portfolio.
prices = market_prices.loc[str(start_date):str(end_date)].copy()
prices = prices.replace([np.inf, -np.inf], np.nan)
prices = prices.where(prices > 0)
prices = prices.dropna(how="any")

if len(prices) < 2:
    st.error(
        "There are not enough common price observations for all selected assets "
        "within the chosen investment period. Select a later start date or a longer period."
    )
    st.stop()

actual_start_date = prices.index[0].date()
actual_end_date = prices.index[-1].date()

st.session_state.portfolio_prices = prices

st.info(
    f"Portfolio calculations use the common available trading history "
    f"from {actual_start_date} to {actual_end_date}. "
    f"You requested {start_date} to {end_date}."
)


# --------------------------------------------------
# PORTFOLIO CONSTRUCTION
# --------------------------------------------------

weights = pd.Series(portfolio_weights) / 100
weights = weights.reindex(prices.columns)

portfolio_returns, allocation_drift = calculate_portfolio_returns(
    prices=prices,
    weights=weights,
    rebalance_frequency=rebalance_frequency
)


# --------------------------------------------------
# PORTFOLIO VALUATION
# --------------------------------------------------

portfolio_value = calculate_portfolio_value(
    portfolio_returns=portfolio_returns,
    initial_investment=initial_investment,
    initial_date=prices.index[0]
)


# --------------------------------------------------
# PORTFOLIO METRICS
# --------------------------------------------------

portfolio_metrics = calculate_portfolio_metrics(
    portfolio_value=portfolio_value,
    risk_free_rate=0.02
)


# --------------------------------------------------
# SAVE PORTFOLIO OUTPUTS TO SHARED STATE
# --------------------------------------------------

st.session_state.portfolio_prices = prices
st.session_state.portfolio_quality_report = quality_report
st.session_state.portfolio_start_date = prices.index[0]
st.session_state.portfolio_end_date = prices.index[-1]
st.session_state.portfolio_returns = portfolio_returns
st.session_state.allocation_drift = allocation_drift
st.session_state.portfolio_value = portfolio_value
st.session_state.portfolio_metrics = portfolio_metrics
st.session_state.portfolio_weights_decimal = weights
st.session_state.portfolio_last_updated = pd.Timestamp.now()


# ==================================================
# 1. PORTFOLIO PERFORMANCE
# ==================================================

st.divider()
st.subheader("Portfolio Performance")

metrics_col, chart_col = st.columns([2, 3], gap="medium")


# --------------------------------------------------
# LEFT: PORTFOLIO METRICS
# --------------------------------------------------

with metrics_col:

    with st.container(
        height=430,
        vertical_alignment="distribute",
        gap="medium"
    ):

        c1, c2 = st.columns(2)
        c1.metric("Portfolio Value", f"${portfolio_value.iloc[-1]:,.2f}")
        c2.metric("Total Return", f"{portfolio_metrics['Total Return']:.2%}")

        c1, c2 = st.columns(2)
        c1.metric("CAGR", f"{portfolio_metrics['CAGR']:.2%}")
        c2.metric(
            "Annualized Volatility",
            f"{portfolio_metrics['Annualized Volatility']:.2%}"
        )

        c1, c2 = st.columns(2)
        c1.metric("Sharpe Ratio", f"{portfolio_metrics['Sharpe Ratio']:.2f}")


# --------------------------------------------------
# RIGHT: PERFORMANCE CHART
# --------------------------------------------------

with chart_col:

    st.markdown("**Portfolio vs. Individual Assets**")

    st.caption(
        f"Compares the growth of ${initial_investment:,.0f} invested "
        "entirely in each individual stock against the same initial capital "
        "distributed across your portfolio using the selected asset weights."
    )

    fig_performance = plot_portfolio_vs_assets(
        prices=prices,
        portfolio_value=portfolio_value
    )

    fig_performance = apply_asset_colors(
        fig_performance,
        asset_colors,
        portfolio_color
    )

    fig_performance.update_layout(
        title=None,
        height=430,
        margin=dict(t=15, b=35, l=35, r=15)
    )

    st.plotly_chart(fig_performance, use_container_width=True)


# ==================================================
# 2. ASSET ALLOCATION
# ==================================================

st.divider()
st.subheader("Asset Allocation")

st.caption(
    "Analyze how your portfolio composition evolves over time "
    "and examine the latest allocation across individual assets."
)

allocation_col, current_col = st.columns([3, 2], gap="medium")


# --------------------------------------------------
# LEFT: HISTORICAL ALLOCATION
# --------------------------------------------------

with allocation_col:

    st.markdown("**Allocation Over Time**")

    allocation_explanations = {
        "Buy and Hold": (
            "Initial target weights are applied when the portfolio is "
            "constructed. No subsequent rebalancing takes place, allowing "
            "asset weights to drift as their market values change."
        ),
        "Monthly": (
            "The portfolio is rebalanced to its target weights on the first "
            "available trading day of each month. Allocation weights are "
            "allowed to drift between rebalancing dates."
        ),
        "Daily": (
            "The portfolio is rebalanced to its target weights every trading "
            "day. Allocation therefore remains close to the original target "
            "weights throughout the investment period."
        )
    }

    st.caption(allocation_explanations[rebalance_frequency])

    st.caption(
        "Each colored band represents an asset's percentage of total "
        "portfolio value. Combined allocation always equals 100%."
    )

    fig_allocation = plot_allocation_over_time(
        allocation_drift=allocation_drift
    )

    fig_allocation = apply_asset_colors(
        fig_allocation,
        asset_colors
    )

    fig_allocation.update_layout(
        title=None,
        height=330,
        margin=dict(t=10, b=35, l=35, r=10)
    )

    st.plotly_chart(fig_allocation, use_container_width=True)


# --------------------------------------------------
# RIGHT: CURRENT ALLOCATION
# --------------------------------------------------

with current_col:

    st.markdown("**Current Portfolio Allocation**")

    st.caption(
        "Shows the latest closing portfolio weights based on the "
        "market values of the selected assets."
    )

    daily_asset_returns = prices.pct_change(fill_method=None).iloc[-1]

    current_weights = allocation_drift.iloc[-1] * (1 + daily_asset_returns)
    current_weights = current_weights / current_weights.sum()

    current_allocation = pd.DataFrame(
        [current_weights],
        index=[prices.index[-1]]
    )

    fig_current = plot_current_allocation(
        allocation_drift=current_allocation
    )

    fig_current = apply_asset_colors(
        fig_current,
        asset_colors
    )

    fig_current.update_layout(
        title=None,
        height=330,
        margin=dict(t=10, b=35, l=10, r=10)
    )

    st.plotly_chart(fig_current, use_container_width=True)
