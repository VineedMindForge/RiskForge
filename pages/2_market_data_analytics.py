import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.market_data import download_data

from src.market_data_analytics import (
    calculate_asset_statistics,
    calculate_asset_returns,
    calculate_return_statistics,
    calculate_rolling_vol,
    calculate_ewma_vol,
    calculate_garch_vol,
    vol_analysis,
    calculate_correlation_matrix,
    calculate_rolling_correlation,
    calculate_normality_test
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Market Data & Analytics",
    page_icon="📈",
    layout="wide"
)

st.title("Market Data & Analytics")

st.markdown(
    "Analyze asset performance, return distributions, volatility dynamics, correlations, "
    "and statistical properties to understand market behaviour and assess the assumptions underlying risk models."
)

st.divider()

# --------------------------------------------------
# CHECK PORTFOLIO CONFIGURATION
# --------------------------------------------------

required_state = [
    "portfolio_prices",
    "selected_assets",
    "portfolio_returns"
]

if any(key not in st.session_state for key in required_state):

    st.warning(
        "Portfolio not configured. Please configure your assets "
        "and portfolio settings in the Portfolio Construction page."
    )

    st.stop()

# --------------------------------------------------
# LOAD PORTFOLIO DATA
# --------------------------------------------------

prices = st.session_state.get("portfolio_prices")
selected_assets = st.session_state.get("selected_assets", [])

if prices is None or prices.empty:
    st.warning("Construct a portfolio on the Portfolio Construction page first.")
    st.stop()

returns = prices.pct_change(fill_method=None).dropna()

# --------------------------------------------------
# SHARED MARKET DATA
# --------------------------------------------------

selected_assets = st.session_state.selected_assets
start_date = st.session_state.start_date
end_date = st.session_state.end_date

prices = st.session_state.get("prices")
quality_report = st.session_state.get("quality_report")
has_warnings = st.session_state.get("has_warnings", False)

if prices is None:
    st.warning("Market data is not available. Visit Portfolio Construction first.")
    st.stop()


# --------------------------------------------------
# ANALYSIS LOOKBACK
# --------------------------------------------------

st.sidebar.header("Analysis Settings")

lookback_options = {
    "30 Days": 30,
    "90 Days": 90,
    "180 Days": 180,
    "1 Year": 252,
    "2 Years": 504,
    "All Available Data": None
}

lookback_label = st.sidebar.selectbox(
    "Historical Analysis Period",
    options=list(lookback_options.keys()),
    index=5
)

lookback = lookback_options[lookback_label]

if lookback is not None:
    analysis_prices = prices.iloc[-lookback:]
else:
    analysis_prices = prices.copy()

analysis_returns = analysis_prices.pct_change(fill_method=None).dropna()



# ==================================================
# 1. NORMALIZED PRICE PERFORMANCE
# ==================================================

st.subheader("Price Performance")
st.caption(
    "Displays the normalized price performance of selected assets over the chosen historical period. "
    "All assets are rebased to 100 at the beginning of the period, allowing their relative performance to be compared."
)

normalized_prices = (
    analysis_prices / analysis_prices.iloc[0]
) * 100

asset_colors = {
    asset: px.colors.qualitative.Dark24[
        i % len(px.colors.qualitative.Dark24)
    ]
    for i, asset in enumerate(st.session_state.available_assets)
}

asset_columns = st.columns(len(selected_assets), gap="medium")

y_min = normalized_prices.min().min()
y_max = normalized_prices.max().max()
y_padding = (y_max - y_min) * 0.05

for col, asset in zip(asset_columns, selected_assets):

    with col:

        fig = px.line(
            normalized_prices,
            y=asset,
            color_discrete_sequence=[asset_colors[asset]]
        )

        fig.update_layout(
            title=asset,
            showlegend=False,
            height=300,
            margin=dict(t=40, b=30, l=10, r=10),
            xaxis_title=None,
            yaxis_title=None,
            yaxis=dict(
                range=[y_min - y_padding, y_max + y_padding]
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"price_performance_{asset}"
        )

# ==================================================
# 2. RETURN DISTRIBUTION
# ==================================================

from scipy.stats import norm

st.divider()
st.subheader("Return Distribution")
st.caption(
    "Daily return distributions over the selected historical period, compared against fitted normal distributions (red curves). "
    "This comparison helps identify deviations from normality, including skewness and heavier tails."
)

asset_columns = st.columns(len(selected_assets), gap="medium")

x_min = analysis_returns.min().min()
x_max = analysis_returns.max().max()

for col, asset in zip(asset_columns, selected_assets):

    with col:

        asset_returns = analysis_returns[asset].dropna()

        mu = asset_returns.mean()
        sigma = asset_returns.std()

        x = np.linspace(x_min, x_max, 500)

        normal_curve = norm.pdf(x, mu, sigma)

        fig = px.histogram(
            asset_returns,
            nbins=100,
            histnorm="probability density",
            color_discrete_sequence=[asset_colors[asset]]
        )

        fig.add_scatter(
            x=x,
            y=normal_curve,
            mode="lines",
            name="Normal Distribution",
            line=dict(color="#FF3030", width=2)
        )

        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="gray"
        )

        fig.update_layout(
            title=asset,
            showlegend=False,
            height=320,
            margin=dict(t=40, b=30, l=10, r=10),
            xaxis_title=None,
            yaxis_title=None,
            xaxis=dict(range=[x_min, x_max])
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"return_distribution_{asset}"
        )

# --------------------------------------------------
# VOLATILITY ANALYSIS
# --------------------------------------------------

st.divider()
st.subheader("Volatility Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    volatility_view = st.selectbox(
        "Select volatility view",
        options=["Asset-wise comparison", "Model-wise comparison"]
    )

with col2:
    volatility_types = st.multiselect(
        "Select volatility models",
        options=["Rolling", "EWMA", "GARCH"],
        default=["Rolling", "EWMA", "GARCH"]
    )

with col3:
    selected_vol_assets = st.multiselect(
        "Select assets",
        options=selected_assets,
        default=selected_assets
    )

if not volatility_types:
    st.warning("Select at least one volatility model.")
    st.stop()

if not selected_vol_assets:
    st.warning("Select at least one asset.")
    st.stop()

daily_returns = calculate_asset_returns(prices[selected_vol_assets])

vol_stats, annual_asset_vol = vol_analysis(daily_returns,window=30,lambda_=0.94)

# ==================================================
# 3. VOLATILITY ANALYSIS
# ==================================================

if volatility_view == "Asset-wise comparison":

    cols = st.columns(2)

    for i, ticker in enumerate(selected_vol_assets):

        with cols[i % 2]:
            asset_vol = vol_stats[ticker]

            fig = go.Figure()

            if "Rolling" in volatility_types:
                fig.add_trace(go.Scatter(
                    x=asset_vol.index,
                    y=asset_vol["rolling_vol"],
                    name="Rolling",
                    line=dict(color="#1f77b4")
                ))

            if "EWMA" in volatility_types:
                fig.add_trace(go.Scatter(
                    x=asset_vol.index,
                    y=asset_vol["ewma_vol"],
                    name="EWMA",
                    line=dict(color="#ff7f0e")
                ))

            if "GARCH" in volatility_types:
                fig.add_trace(go.Scatter(
                    x=asset_vol.index,
                    y=asset_vol["garch_vol"],
                    name="GARCH",
                    line=dict(color="#2ca02c")
                ))

            fig.add_hline(
                y=annual_asset_vol[ticker],
                line_dash="dash",
                annotation_text="Historical volatility"
            )

            fig.update_layout(
                template="plotly_white",
                hovermode="x unified",
                title=f"{ticker} — Annualised Volatility",
                xaxis_title="Date",
                yaxis_title="Annualised Volatility",
                legend_title="Volatility Model",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

else:

    volatility_columns = {
        "Rolling": "rolling_vol",
        "EWMA": "ewma_vol",
        "GARCH": "garch_vol"
    }

    cols = st.columns(2)

    for i, model in enumerate(volatility_types):

        with cols[i % 2]:
            fig = go.Figure()

            for ticker in selected_vol_assets:
                asset_vol = vol_stats[ticker]

                fig.add_trace(go.Scatter(
                    x=asset_vol.index,
                    y=asset_vol[volatility_columns[model]],
                    name=ticker
                ))

            fig.update_layout(
                template="plotly_white",
                hovermode="x unified",
                title=f"{model} Volatility — Asset Comparison",
                xaxis_title="Date",
                yaxis_title="Annualised Volatility",
                legend_title="Asset",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# CORRELATION ANALYSIS
# --------------------------------------------------

st.divider()
st.subheader("Correlation Analysis")
st.caption(
    "The correlation matrix shows pairwise correlations across the entire selected "
    "historical period. Rolling correlation tracks the relationship between two "
    "selected assets over a user-defined lookback window. "
    "Lower or negative correlations generally provide greater diversification "
    "benefits, as assets are less likely to move together. However, correlations "
    "can change over time, particularly during periods of market stress."
)
correlation_matrix = calculate_correlation_matrix(daily_returns)

fig_corr = px.imshow(
    correlation_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    aspect="equal",
    title="Asset Correlation Matrix"
)
corr_col1, corr_col2 = st.columns(2)

with corr_col1:
    st.markdown("**Correlation Matrix**")

    fig_corr.update_layout(
        template="plotly_white",
        width=600,
        height=600
    )

    st.plotly_chart(fig_corr, use_container_width=True)

with corr_col2:
    st.markdown("**Rolling Correlation**")

    col1, col2, col3 = st.columns(3)

    with col1:
        asset1 = st.selectbox(
            "First asset",
            options=selected_vol_assets,
            index=0
        )

    with col2:
        asset2 = st.selectbox(
            "Second asset",
            options=selected_vol_assets,
            index=1 if len(selected_vol_assets) > 1 else 0
        )

    with col3:
        correlation_window = st.selectbox(
            "Lookback window (trading days)",
            options=[30, 60, 90, 120],
            index=1
        )

    if asset1 == asset2:
        st.warning("Select two different assets to calculate rolling correlation.")

    else:
        rolling_corr = calculate_rolling_correlation(
            daily_returns,
            asset1,
            asset2,
            window=correlation_window
        )

        fig_rolling_corr = go.Figure()

        fig_rolling_corr.add_trace(go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr,
            name=f"{asset1} vs {asset2}",
            line=dict(color="#1f77b4")
        ))

        fig_rolling_corr.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray"
        )

        fig_rolling_corr.update_layout(
            template="plotly_white",
            title=f"{asset1} vs {asset2}",
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Correlation",
            yaxis_range=[-1, 1],
            height=500
        )

        st.plotly_chart(fig_rolling_corr, use_container_width=True)

# --------------------------------------------------
# RETURN DISTRIBUTION DIAGNOSTICS
# --------------------------------------------------

st.divider()
st.subheader("Return Distribution Diagnostics")

st.caption(
    "Assess the shape and statistical properties of asset returns. "
    "Skewness measures asymmetry, excess kurtosis measures tail heaviness, "
    "and the Jarque–Bera test evaluates the normality assumption."
)

distribution_stats = pd.DataFrame({
    "Skewness": daily_returns.skew(),
    "Excess Kurtosis": daily_returns.kurtosis()
})

normality_results = calculate_normality_test(daily_returns)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Skewness**")

    fig_skew = px.bar(
        distribution_stats.reset_index(),
        x="Ticker",
        y="Skewness",
        text_auto=".3f",
        labels={"Ticker": "Asset", "Skewness": "Skewness"}
    )

    fig_skew.add_hline(y=0, line_dash="dash", line_color="gray")

    fig_skew.update_layout(
        template="plotly_white",
        showlegend=False,
        height=350
    )

    st.plotly_chart(fig_skew, use_container_width=True)

with col2:
    st.markdown("**Excess Kurtosis**")

    fig_kurt = px.bar(
        distribution_stats.reset_index(),
        x="Ticker",
        y="Excess Kurtosis",
        text_auto=".3f",
        labels={"Ticker": "Asset", "Excess Kurtosis": "Excess Kurtosis"}
    )

    fig_kurt.add_hline(y=0, line_dash="dash", line_color="gray")

    fig_kurt.update_layout(
        template="plotly_white",
        showlegend=False,
        height=350
    )

    st.plotly_chart(fig_kurt, use_container_width=True)

st.markdown("**Jarque–Bera Normality Test**")

normality_display = normality_results[
    ["observations", "jb_statistic", "p_value", "normality_not_rejected"]
].copy()

normality_display["Normality Assessment"] = normality_display["normality_not_rejected"].map({
    True: "Not Rejected",
    False: "Rejected"
})

normality_display = normality_display.drop(columns="normality_not_rejected")

normality_display = normality_display.rename(columns={
    "observations": "Observations",
    "jb_statistic": "JB Statistic",
    "p_value": "P-Value"
})

st.dataframe(
    normality_display.style.format({
        "JB Statistic": "{:.2f}",
        "P-Value": "{:.4g}"
    }),
    use_container_width=True
)

st.caption(
    "H₀: Returns follow a normal distribution. "
    "P-value < 0.05 indicates rejection of the normality assumption."
)