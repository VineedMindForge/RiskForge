"""VaR forecasts and statistical backtesting functions."""

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from scipy.special import xlogy


def return_historical_var_series(portfolio_returns, window_size=250, confidence_level=0.95):
    """Create one-day-ahead Historical Simulation VaR forecasts."""

    alpha = 1 - confidence_level
    return_threshold = portfolio_returns.rolling(window_size).quantile(alpha).shift(1)
    return -return_threshold


def return_parametric_var_series(portfolio_returns, window_size=250, confidence_level=0.95):
    """Create one-day-ahead normal-distribution VaR forecasts."""

    alpha = 1 - confidence_level
    rolling_mean = portfolio_returns.rolling(window_size).mean()
    rolling_volatility = portfolio_returns.rolling(window_size).std()
    return_threshold = (rolling_mean + norm.ppf(alpha) * rolling_volatility).shift(1)
    return -return_threshold


def return_ewma_var_series(portfolio_returns, lambda_=0.94, window_size=250, confidence_level=0.95):
    """Create one-day-ahead VaR forecasts using EWMA volatility."""

    alpha = 1 - confidence_level
    ewma_volatility = portfolio_returns.pow(2).ewm(
        alpha=1 - lambda_,
        min_periods=window_size,
        adjust=False
    ).mean().pow(0.5)

    rolling_mean = portfolio_returns.rolling(window_size).mean()
    return_threshold = (rolling_mean + norm.ppf(alpha) * ewma_volatility).shift(1)
    return -return_threshold


def return_garch_var_series(portfolio_returns, window_size=250, confidence_level=0.95):
    """Create one-day-ahead GARCH(1,1) VaR forecasts.

    Parameters are fitted using the initial window and then held fixed. The
    conditional variance is updated each day using the latest observed return.
    """

    try:
        from arch import arch_model
    except ImportError as error:
        raise ImportError("GARCH estimation requires the arch package. Install it with: pip install arch") from error

    if window_size < 50:
        raise ValueError("Use at least 50 observations to estimate GARCH.")

    clean_returns = portfolio_returns.dropna()

    if len(clean_returns) <= window_size:
        raise ValueError("Provide more observations than the GARCH estimation window.")

    if not np.isfinite(clean_returns.to_numpy()).all():
        raise ValueError("portfolio_returns must contain finite values.")

    alpha = 1 - confidence_level
    var_series = pd.Series(np.nan, index=portfolio_returns.index, dtype=float)

    training_returns = clean_returns.iloc[:window_size] * 100
    model = arch_model(training_returns, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    fitted_model = model.fit(disp="off")

    parameters = fitted_model.params
    mean_return = parameters["mu"]
    omega = parameters["omega"]
    alpha_garch = parameters["alpha[1]"]
    beta_garch = parameters["beta[1]"]
    conditional_variance = fitted_model.conditional_volatility.iloc[-1] ** 2

    for position in range(window_size, len(clean_returns)):
        previous_return = clean_returns.iloc[position - 1] * 100
        residual = previous_return - mean_return
        conditional_variance = omega + alpha_garch * residual ** 2 + beta_garch * conditional_variance
        forecast_volatility = np.sqrt(conditional_variance)
        return_threshold = mean_return + norm.ppf(alpha) * forecast_volatility
        date = clean_returns.index[position]
        var_series.loc[date] = -return_threshold / 100

    return var_series


def calculate_var_exceptions(portfolio_returns, var_series):
    """Align realised returns with forecasts and identify VaR exceptions."""

    aligned_data = pd.concat(
        [portfolio_returns.rename("returns"), var_series.rename("var")],
        axis=1
    ).dropna()

    if aligned_data.empty:
        raise ValueError("There are no overlapping return and VaR forecast observations.")

    exceptions = aligned_data["returns"] < -aligned_data["var"]

    return {
        "exceptions": exceptions,
        "exceptions_count": int(exceptions.sum()),
        "exceptions_rate": float(exceptions.mean())
    }


def kupiec_test(exception_count, observations, confidence_level=0.95):
    """Run the Kupiec unconditional-coverage test."""

    if observations < 1 or exception_count < 0 or exception_count > observations:
        raise ValueError("exception_count and observations are not valid.")

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    alpha = 1 - confidence_level
    exception_rate = exception_count / observations

    log_likelihood_null = xlogy(observations - exception_count, 1 - alpha) + xlogy(exception_count, alpha)
    log_likelihood_alternative = xlogy(observations - exception_count, 1 - exception_rate) + xlogy(exception_count, exception_rate)

    lr_statistic = max(0, -2 * (log_likelihood_null - log_likelihood_alternative))
    p_value = chi2.sf(lr_statistic, df=1)

    return {
        "lr_statistic": lr_statistic,
        "p_value": p_value,
        "reject_null": p_value < 0.05,
        "observed_rate": exception_rate,
        "expected_rate": alpha
    }


def christoffersen_test(exceptions):
    """Run the Christoffersen independence test for clustered exceptions."""

    exceptions = pd.Series(exceptions)

    if len(exceptions) < 2:
        raise ValueError("At least two exception observations are required.")

    if exceptions.isna().any():
        raise ValueError("Exceptions cannot contain missing values.")

    if not exceptions.isin([0, 1, True, False]).all():
        raise ValueError("Exceptions must contain only binary values (0 or 1).")

    previous = exceptions.iloc[:-1].astype(int).to_numpy()
    current = exceptions.iloc[1:].astype(int).to_numpy()

    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())

    total_transitions = n00 + n01 + n10 + n11

    probability = (n01 + n11) / total_transitions
    probability_0 = n01 / (n00 + n01) if n00 + n01 > 0 else 0
    probability_1 = n11 / (n10 + n11) if n10 + n11 > 0 else 0

    log_likelihood_independent = (
        xlogy(n00 + n10, 1 - probability) +
        xlogy(n01 + n11, probability)
    )

    log_likelihood_markov = (
        xlogy(n00, 1 - probability_0) +
        xlogy(n01, probability_0) +
        xlogy(n10, 1 - probability_1) +
        xlogy(n11, probability_1)
    )

    lr_statistic = max(0, -2 * (log_likelihood_independent - log_likelihood_markov))
    p_value = chi2.sf(lr_statistic, df=1)

    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "pi_0": probability_0,
        "pi_1": probability_1,
        "lr_statistic": lr_statistic,
        "p_value": p_value,
        "reject_null": p_value < 0.05
    }

def conditional_coverage_test(kupiec_result, christoffersen_result):
    """Combine Kupiec and independence statistics in the conditional-coverage test."""

    lr_statistic = kupiec_result["lr_statistic"] + christoffersen_result["lr_statistic"]
    p_value = chi2.sf(lr_statistic, df=2)

    return {
        "lr_statistic": lr_statistic,
        "p_value": p_value,
        "reject_null": p_value < 0.05
    }


def backtest_var(portfolio_returns, method="historical", window_size=250, confidence_level=0.95, lambda_=0.94,  evaluation_start=None):
    """Generate a VaR series and calculate its backtesting statistics."""

    if method == "historical":
        var_series = return_historical_var_series(portfolio_returns, window_size, confidence_level)
    elif method == "parametric":
        var_series = return_parametric_var_series(portfolio_returns, window_size, confidence_level)
    elif method == "ewma":
        var_series = return_ewma_var_series(portfolio_returns, lambda_, window_size, confidence_level)
    elif method == "garch":
        var_series = return_garch_var_series(portfolio_returns, window_size, confidence_level)
    else:
        raise ValueError("method must be historical, parametric, ewma or garch.")

    if evaluation_start is not None:
        var_series = var_series.loc[evaluation_start:]

    exception_results = calculate_var_exceptions(portfolio_returns, var_series)
    exceptions = exception_results["exceptions"]
    exception_count = exception_results["exceptions_count"]
    exception_rate = exception_results["exceptions_rate"]
    observations = len(exceptions)

    kupiec_results = kupiec_test(exception_count, observations, confidence_level)
    christoffersen_results = christoffersen_test(exceptions)
    coverage_results = conditional_coverage_test(kupiec_results, christoffersen_results)

    return {
        "method": method,
        "var_forecast": var_series,
        "exceptions": exceptions,
        "exception_count": exception_count,
        "exception_rate": exception_rate,
        "observations": observations,
        "kupiec": kupiec_results,
        "christoffersen": christoffersen_results,
        "conditional_coverage": coverage_results
    }



def compare_var_models(portfolio_returns, window_size=250, confidence_level=0.95, lambda_=0.94, evaluation_start=None):
    """Run the four VaR methods and return a comparison table."""

    methods = ["historical", "parametric", "ewma", "garch"]
    comparison = []

    for method in methods:

        results = backtest_var(
            portfolio_returns=portfolio_returns,
            method=method,
            window_size=window_size,
            confidence_level=confidence_level,
            lambda_=lambda_,
            evaluation_start=evaluation_start
        )

        comparison.append({
            "Method": method,
            "Observations": results["observations"],
            "Exceptions": results["exception_count"],
            "Exception Rate": results["exception_rate"],
            "Expected Rate": results["kupiec"]["expected_rate"],
            "Kupiec p-value": results["kupiec"]["p_value"],
            "Independence p-value": results["christoffersen"]["p_value"],
            "Conditional Coverage p-value": results["conditional_coverage"]["p_value"]
        })

    return pd.DataFrame(comparison).set_index("Method")

def compare_backtesting_tests(model_results):
    """Compare statistical VaR backtesting tests across models."""

    results = []

    for method, result in model_results.items():
        conditional_p = result["conditional_coverage"]["p_value"]

        results.append({
            "Model": method.title(),
            "Kupiec p-value": result["kupiec"]["p_value"],
            "Independence p-value": result["christoffersen"]["p_value"],
            "Conditional Coverage p-value": conditional_p,
            "Overall Assessment": "Not Rejected" if conditional_p >= 0.05 else "Rejected"
        })

    return pd.DataFrame(results).set_index("Model")

def plot_var_backtesting(portfolio_returns, model_result, method):
    """Plot realised returns against VaR forecasts and highlight exceptions."""

    import plotly.graph_objects as go

    var_series = model_result["var_forecast"]
    exceptions = model_result["exceptions"]
    exception_rate = model_result["exception_rate"]
    exception_count = model_result["exception_count"]
    observations = model_result["observations"]

    aligned_data = pd.concat(
        [
            portfolio_returns.rename("Returns"),
            var_series.rename("VaR"),
            exceptions.rename("Exception")
        ],
        axis=1
    ).dropna()

    fig = go.Figure()

    # Actual portfolio returns
    fig.add_trace(
        go.Scatter(
            x=aligned_data.index,
            y=aligned_data["Returns"],
            mode="lines",
            name="Realised Returns",
            line=dict(color="#636EFA", width=1)
        )
    )

    # VaR threshold
    fig.add_trace(
        go.Scatter(
            x=aligned_data.index,
            y=-aligned_data["VaR"],
            mode="lines",
            name="VaR Threshold",
            line=dict(color="#EF553B", width=2, dash="dash")
        )
    )

    # VaR exceptions
    exception_data = aligned_data[
        aligned_data["Exception"]
    ]

    fig.add_trace(
        go.Scatter(
            x=exception_data.index,
            y=exception_data["Returns"],
            mode="markers",
            name="VaR Exceptions",
            marker=dict(
                color="#EF553B",
                size=8,
                symbol="circle"
            )
        )
    )

    fig.update_layout(
        title=f"{method} VaR Backtesting",
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Daily Return",
        yaxis_tickformat=".1%",
        hovermode="x unified",
        height=500,
        legend_title=None
    )

    fig.add_annotation(
    text=(
        f"Exception Rate: {exception_rate:.2%} "
        f"({exception_count}/{observations})"
    ),
    xref="paper",
    yref="paper",
    x=0.99,
    y=1.12,
    showarrow=False,
    xanchor="right",
    font=dict(size=13)
    )

    return fig

def plot_exception_rate_comparison(model_results, confidence_level):
    """Compare actual and expected VaR exception rates across models."""

    import plotly.graph_objects as go

    methods = list(model_results.keys())
    actual_rates = [model_results[m]["exception_rate"] * 100 for m in methods]
    expected_rate = (1 - confidence_level) * 100

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[m.title() for m in methods],
        y=actual_rates,
        name="Actual Exception Rate"
    ))

    fig.add_trace(go.Bar(
        x=[m.title() for m in methods],
        y=[expected_rate] * len(methods),
        name="Expected Exception Rate"
    ))

    fig.update_layout(
        title="Actual vs Expected Exception Rate",
        template="plotly_white",
        xaxis_title="VaR Model",
        yaxis_title="Exception Rate (%)",
        barmode="group",
        height=400,
        legend_title=None
    )

    return fig

def plot_rolling_exception_rate(model_results, confidence_level, window=60, selected_methods=None):
    """Plot rolling VaR exception rates across selected models."""

    import plotly.graph_objects as go

    if selected_methods is None:
        selected_methods = list(model_results.keys())

    fig = go.Figure()

    for method in selected_methods:
        exceptions = model_results[method]["exceptions"]
        rolling_rate = exceptions.rolling(window).mean() * 100

        fig.add_trace(go.Scatter(
            x=rolling_rate.index,
            y=rolling_rate,
            mode="lines",
            name=method.title()
        ))

    expected_rate = (1 - confidence_level) * 100

    fig.add_hline(
        y=expected_rate,
        line_dash="dash",
        line_color="#F2C94C",
        line_width=2,
        annotation_text=f"Expected Exception Rate: {expected_rate:.1f}%",
        annotation_position="top left",
        annotation_font=dict(color="white", size=11),
        annotation_bgcolor="#333333",
        annotation_bordercolor="#F2C94C",
        annotation_borderpad=4
    )

    fig.update_layout(
        title=f"Rolling Exception Rate ({window}-Day Window)",
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Exception Rate (%)",
        height=450,
        hovermode="x unified",
        legend_title=None
    )

    return fig
