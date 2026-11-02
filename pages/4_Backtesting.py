
import streamlit as st
import pandas as pd
import numpy as np

from src.backtesting import backtest_var, compare_var_models, plot_var_backtesting, plot_exception_rate_comparison, plot_rolling_exception_rate, compare_backtesting_tests


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Backtesting & Model Validation",
    page_icon="📈",
    layout="wide"
)

st.title("Backtesting & Model Validation")

st.markdown(
    "Evaluate the accuracy and reliability of portfolio VaR forecasts "
    "against realised market returns."
)

st.divider()


# --------------------------------------------------
# CHECK PORTFOLIO DATA
# --------------------------------------------------

if st.session_state.get("portfolio_returns") is None:
    st.warning(
        "Construct a portfolio on the Portfolio Construction page first."
    )
    st.stop()


# --------------------------------------------------
# RETRIEVE PORTFOLIO DATA
# --------------------------------------------------

portfolio_returns = st.session_state.portfolio_returns.dropna()


# --------------------------------------------------
# DATA VALIDATION
# --------------------------------------------------

window_options = [250, 500, 750]
max_estimation_window = max(window_options)

if len(portfolio_returns) <= max_estimation_window:
    st.error(
        f"At least {max_estimation_window + 1} observations "
        "are required for a common backtesting period."
    )
    st.stop()

if not np.isfinite(portfolio_returns.to_numpy()).all():
    st.error("Portfolio returns contain invalid values.")
    st.stop()


# --------------------------------------------------
# SIDEBAR: BACKTESTING SETTINGS
# --------------------------------------------------

st.sidebar.header("Backtesting Settings")

confidence_level = st.sidebar.selectbox(
    "Confidence Level",
    options=[0.95, 0.99],
    format_func=lambda x: f"{x:.0%}",
    index=0
)

window_size = st.sidebar.selectbox(
    "Estimation Window",
    options=window_options,
    index=0,
    help="Number of historical daily returns used to estimate VaR."
)

st.sidebar.subheader("Evaluation Period")

evaluation_mode = st.sidebar.radio(
    "Evaluation period",
    options=["Automatic", "Custom"],
    help="Choose the period over which VaR forecasts are evaluated."
)

# Earliest valid date depends on the selected estimation window.
earliest_date = portfolio_returns.index[window_size]

if evaluation_mode == "Automatic":
    evaluation_start = portfolio_returns.index[max_estimation_window]

else:
    evaluation_date = st.sidebar.date_input(
        "Evaluation start date",
        value=earliest_date.date(),
        min_value=earliest_date.date(),
        max_value=portfolio_returns.index[-1].date()
    )

    evaluation_start = portfolio_returns.index[
        portfolio_returns.index >= pd.Timestamp(evaluation_date)
    ][0]

lambda_ = st.sidebar.number_input(
    "EWMA Lambda",
    min_value=0.80,
    max_value=0.99,
    value=0.94,
    step=0.01,
    help="Decay factor used in the EWMA volatility calculation."
)


# --------------------------------------------------
# BACKTESTING MODEL CALCULATIONS
# --------------------------------------------------

st.subheader("Running Backtesting Models")

with st.spinner("Generating VaR forecasts and running statistical tests..."):

    try:
        model_results = {}

        for method in ["historical", "parametric", "ewma", "garch"]:

            model_results[method] = backtest_var(
                portfolio_returns=portfolio_returns,
                method=method,
                window_size=window_size,
                confidence_level=confidence_level,
                lambda_=lambda_,
                evaluation_start=evaluation_start
            )

    except (ValueError, ImportError) as error:
        st.error(f"Backtesting failed: {error}")
        st.stop()

st.success("All four VaR models completed successfully.")

st.caption(
    f"Evaluation period: {evaluation_start:%Y-%m-%d} onwards. "
    f"Estimation window: {window_size} trading days."
)


# --------------------------------------------------
# VAR FORECAST VS REALISED RETURNS
# --------------------------------------------------

st.divider()

st.subheader("VaR Forecast vs Realised Returns")

st.markdown(
    "The evaluation period defines the dates over which realised returns are compared against VaR forecasts. "
    "VaR is estimated using the selected estimation window of historical returns preceding each evaluation date. "
    "Forecasts are generated on a rolling, one-step-ahead basis, ensuring that only information available at the time "
    "of each forecast is used and preventing look-ahead bias (data leakage)."
)

col_metrics, col_chart = st.columns([1, 2])

with col_metrics:
    st.subheader("Backtesting Summary")

    selected_model = st.selectbox(
        "VaR Model",
        options=["historical", "parametric", "ewma", "garch"],
        format_func=lambda x: x.title()
    )

    selected_result = model_results[selected_model]

    observations = selected_result["observations"]
    exception_count = selected_result["exception_count"]
    exception_rate = selected_result["exception_rate"]
    expected_rate = 1 - confidence_level
    rate_difference = exception_rate - expected_rate

    st.markdown("**Evaluation**")
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Observations", f"{observations:,}")
    metric_col2.metric("Exception Count", f"{exception_count:,}")

    st.markdown("**Exception Rates**")
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Actual Exception Rate", f"{exception_rate:.2%}")
    metric_col2.metric("Expected Exception Rate", f"{expected_rate:.2%}")

    st.markdown("**Deviation**")
    st.metric("Exception Rate Difference", f"{rate_difference:+.2%}")

with col_chart:
    st.subheader("VaR Forecast vs Realised Returns")

    fig = plot_var_backtesting(
        portfolio_returns,
        selected_result,
        selected_model.title()
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Actual vs Expected Exception Rate
# --------------------------------------------------
st.divider()
st.subheader("Exception Analysis")

fig = plot_exception_rate_comparison(model_results, confidence_level)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Rolling Exception Rate
# --------------------------------------------------

st.subheader("Rolling Exception Rate")
st.caption(
    "Select the VaR models you want to compare and the rolling window in trading days. "
    "The rolling exception rate measures the percentage of VaR exceptions within the selected "
    "number of preceding trading days. For example, a 20-day window calculates the exception "
    "rate using the most recent 20 trading observations, while a 250-day window uses the most "
    "recent 250 observations. The window moves forward by one trading day at a time. "
    "Larger windows produce smoother trends by averaging exceptions over a longer period, "
    "reducing short-term fluctuations, while smaller windows are more responsive to recent "
    "changes in model performance."
)

filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    selected_methods = st.multiselect(
        "Select models to display",
        options=list(model_results.keys()),
        default=list(model_results.keys()),
        format_func=lambda x: x.title()
    )

with filter_col2:
    rolling_window = st.selectbox(
        "Rolling Window",
        options=[20, 60, 120, 250],
        index=1,
        format_func=lambda x: f"{x} Trading Days"
    )
fig = plot_rolling_exception_rate(
    model_results,
    confidence_level,
    window=rolling_window,
    selected_methods=selected_methods
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# Statistical Model Validation
# --------------------------------------------------

st.divider()
st.subheader("Statistical Model Validation")

st.caption(
    "The tests evaluate whether VaR exceptions occur at the expected frequency "
    "and independently over time. A p-value below 0.05 indicates rejection of "
    "the corresponding null hypothesis."
)

validation_results = compare_backtesting_tests(model_results)

p_value_columns = [
    "Kupiec p-value",
    "Independence p-value",
    "Conditional Coverage p-value"
]

styled_results = (
    validation_results.style
    .format("{:.4f}", subset=p_value_columns)
    .map(
        lambda value: "background-color: #7f1d1d; color: white"
        if value < 0.05 else "background-color: #14532d; color: white",
        subset=p_value_columns
    )
)

st.dataframe(styled_results, use_container_width=True)
st.caption(
    "The tests are evaluated at a 5% significance level. "
    "Green indicates that the null hypothesis is not rejected; "
    "red indicates that it is rejected.")


with st.expander("1. Kupiec Test — Unconditional Coverage"):
    st.markdown("""
    **Purpose:** Tests whether the observed frequency of VaR exceptions
    matches the expected exception frequency.

    **Null Hypothesis (H₀):** The probability of a VaR exception equals the expected tail probability.
    At 95% confidence, the expected exception rate is 5%.

    **Alternative Hypothesis (H₁):** The actual exception probability differs from the expected probability.

    **Interpretation:** A rejection indicates that the model's exception frequency is inconsistent with the selected confidence level.
    """)

with st.expander("2. Christoffersen Test — Independence"):
    st.markdown("""
    **Purpose:** Tests whether VaR exceptions occur independently over time.

    **Null Hypothesis (H₀):** The probability of an exception today does not depend on whether
    yesterday was an exception.

    **Alternative Hypothesis (H₁):** The probability of an exception depends on the previous day's
    exception status.

    **Interpretation:** A rejection indicates evidence of exception clustering, suggesting that the model may not adequately capture changing market risk.
    """)

with st.expander("3. Christoffersen Test — Conditional Coverage"):
    st.markdown("""
    **Purpose:** Combines unconditional coverage and independence into a single test.

    **Null Hypothesis (H₀):** The model produces the expected exception frequency, and exceptions
    occur independently over time.

    **Alternative Hypothesis (H₁):** At least one of these conditions is violated.

    **Calculation:**
    LR(CC) = LR(POF) + LR(IND)

    The test statistic follows a chi-square distribution with 2 degrees of freedom.

    **Interpretation:** Rejection indicates that the model fails at least one of the two coverage requirements.
    """)

with st.expander("How to Interpret P-values"):
    st.markdown("""

    **A higher p-value is generally preferable.** It indicates insufficient
    statistical evidence to reject the null hypothesis.

    However, failing to reject H₀ does not prove that a model is correct.
    """)

