"""Historical, parametric and Monte Carlo Value at Risk calculations."""

import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_holding_period_returns(portfolio_returns, holding_period=1):
    """Calculate overlapping compounded returns over a chosen holding period."""

    if holding_period < 1 or int(holding_period) != holding_period:
        raise ValueError("holding_period must be a positive integer.")

    if portfolio_returns.isna().any():
        raise ValueError("portfolio_returns contains missing values.")

    if not np.isfinite(portfolio_returns.to_numpy()).all():
        raise ValueError("portfolio_returns must contain finite values.")

    if (portfolio_returns <= -1).any():
        raise ValueError("Simple returns cannot be less than or equal to -100%.")

    holding_period_returns = (1 + portfolio_returns).rolling(holding_period).apply(np.prod, raw=True) - 1
    return holding_period_returns.dropna()


def historical_var_es(portfolio_returns, investment_value, confidence_level=0.95, window=None, holding_period=1):
    """Calculate Historical Simulation VaR and Expected Shortfall."""

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    if investment_value <= 0 or not np.isfinite(investment_value):
        raise ValueError("investment_value must be finite and positive.")

    if window is not None:
        if window < 1:
            raise ValueError("window must be at least 1.")
        portfolio_returns = portfolio_returns.tail(window)

    holding_period_returns = calculate_holding_period_returns(portfolio_returns, holding_period)

    if holding_period_returns.empty:
        raise ValueError("Not enough observations for the selected holding period.")

    alpha = 1 - confidence_level
    return_threshold = holding_period_returns.quantile(alpha)
    var_return = -return_threshold
    es_return = -holding_period_returns[holding_period_returns <= return_threshold].mean()

    return {
        "var_return": var_return,
        "var_amount": var_return * investment_value,
        "es_return": es_return,
        "es_amount": es_return * investment_value,
        "observations": len(holding_period_returns),
        "confidence_level": confidence_level,
        "holding_period": holding_period
    }


def parametric_var_es(portfolio_returns, investment_value, confidence_level=0.95, window=None, holding_period=1):
    """Calculate normal-distribution VaR and Expected Shortfall."""

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    if investment_value <= 0 or not np.isfinite(investment_value):
        raise ValueError("investment_value must be finite and positive.")

    if holding_period < 1 or int(holding_period) != holding_period:
        raise ValueError("holding_period must be a positive integer.")

    if window is not None:
        if window < 2:
            raise ValueError("window must be at least 2.")
        portfolio_returns = portfolio_returns.tail(window)

    clean_returns = portfolio_returns.dropna()

    if len(clean_returns) < 2 or not np.isfinite(clean_returns.to_numpy()).all():
        raise ValueError("At least two finite return observations are required.")

    alpha = 1 - confidence_level
    mean_return = clean_returns.mean()
    daily_volatility = clean_returns.std(ddof=1)
    z_score = norm.ppf(alpha)

    var_return = -(mean_return * holding_period + daily_volatility * z_score * np.sqrt(holding_period))
    es_return = -holding_period * mean_return + np.sqrt(holding_period) * daily_volatility * norm.pdf(z_score) / alpha

    return {
        "var_return": var_return,
        "var_amount": var_return * investment_value,
        "es_return": es_return,
        "es_amount": es_return * investment_value,
        "observations": len(clean_returns),
        "confidence_level": confidence_level,
        "holding_period": holding_period
    }


def monte_carlo_var_es(daily_returns, weights, investment_value, confidence_level=0.95, n_simulations=10000, window=None, holding_period=1, random_seed=42):
    """Calculate Monte Carlo VaR and ES using correlated normal asset returns.

    The portfolio is rebalanced to the target weights every simulated day.
    """

    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    if investment_value <= 0 or not np.isfinite(investment_value):
        raise ValueError("investment_value must be finite and positive.")

    if n_simulations < 1 or holding_period < 1:
        raise ValueError("n_simulations and holding_period must be positive.")

    if window is not None:
        if window < 2:
            raise ValueError("window must be at least 2.")
        daily_returns = daily_returns.tail(window)

    daily_returns = daily_returns.dropna(how="any")

    if len(daily_returns) < 2:
        raise ValueError("At least two complete return observations are required.")

    if not np.isfinite(daily_returns.to_numpy()).all() or (daily_returns <= -1).any().any():
        raise ValueError("Asset returns must be finite and greater than -100%.")

    if isinstance(weights, pd.Series):
        weights = weights.reindex(daily_returns.columns).to_numpy(dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    if len(weights) != daily_returns.shape[1] or not np.isfinite(weights).all():
        raise ValueError("Provide one finite weight for each asset.")

    if (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Weights must be non-negative and sum to 1.")

    mean_returns = daily_returns.mean().to_numpy()
    covariance_matrix = daily_returns.cov().to_numpy()
    rng = np.random.default_rng(random_seed)

    simulated_asset_returns = rng.multivariate_normal(
        mean=mean_returns,
        cov=covariance_matrix,
        size=(n_simulations, holding_period)
    )

    simulated_portfolio_returns = simulated_asset_returns @ weights
    holding_period_returns = np.prod(1 + simulated_portfolio_returns, axis=1) - 1

    alpha = 1 - confidence_level
    return_threshold = np.quantile(holding_period_returns, alpha)
    var_return = -return_threshold
    tail_returns = holding_period_returns[holding_period_returns <= return_threshold]
    es_return = -tail_returns.mean()

    return {
        "var_return": var_return,
        "var_amount": var_return * investment_value,
        "es_return": es_return,
        "es_amount": es_return * investment_value,
        "observations": len(holding_period_returns),
        "confidence_level": confidence_level,
        "holding_period": holding_period
    }

def calculate_var_comparison(
    portfolio_returns,
    asset_returns,
    weights,
    investment_value,
    confidence_level=0.95,
    holding_period=1,
    window=None,
    n_simulations=10000
):
    """
    Calculate and compare Historical, Parametric and Monte Carlo VaR and ES.
    """

    historical = historical_var_es(
        portfolio_returns,
        investment_value,
        confidence_level,
        window,
        holding_period
    )

    parametric = parametric_var_es(
        portfolio_returns,
        investment_value,
        confidence_level,
        window,
        holding_period
    )

    monte_carlo = monte_carlo_var_es(
        asset_returns,
        weights,
        investment_value,
        confidence_level,
        n_simulations,
        window,
        holding_period
    )

    comparison = pd.DataFrame({
        "Historical": historical,
        "Parametric": parametric,
        "Monte Carlo": monte_carlo
    }).T

    comparison = comparison[
        ["var_return", "var_amount", "es_return", "es_amount"]
    ]

    comparison.columns = ["VaR (%)", "VaR ($)", "ES (%)", "ES ($)"]

    return comparison



def plot_return_distribution(
    portfolio_returns,
    var_comparison,
    selected_model="Historical",
    holding_period=1
):
    """Plot holding-period returns with the selected model's VaR and ES."""

    import plotly.graph_objects as go

    returns = calculate_holding_period_returns(
        portfolio_returns,
        holding_period
    )

    selected_var = var_comparison.loc[selected_model, "VaR (%)"]
    selected_es = var_comparison.loc[selected_model, "ES (%)"]

    var_threshold = -selected_var
    es_threshold = -selected_es

    fig = go.Figure()

    # Holding-period return distribution
    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=60,
            histnorm="probability density",
            name="Portfolio Returns",
            marker_color="#636EFA",
            opacity=0.8
        )
    )

    # Highlight the selected model's tail
    fig.add_vrect(
        x0=returns.min(),
        x1=var_threshold,
        fillcolor="#EF553B",
        opacity=0.15,
        layer="below",
        line_width=0
    )

    # VaR threshold
    fig.add_vline(
        x=var_threshold,
        line_dash="dash",
        line_width=2,
        line_color="#EF553B"
    )

    fig.add_annotation(
        x=var_threshold,
        y=0.5,
        yref="paper",
        text=f"{selected_model} VaR: {selected_var:.2%}",
        textangle=-90,
        showarrow=False,
        xanchor="right",
        yanchor="middle",
        xshift=0,
        font=dict(color="#EF553B", size=11)
    )

    # Expected Shortfall threshold
    fig.add_vline(
        x=es_threshold,
        line_dash="dash",
        line_width=2,
        line_color="#AB63FA"
    )

    fig.add_annotation(
        x=es_threshold,
        y=0.5,
        yref="paper",
        text=f"{selected_model} ES: {selected_es:.2%}",
        textangle=-90,
        showarrow=False,
        xanchor="right",
        yanchor="middle",
        xshift=0,
        font=dict(color="#AB63FA", size=11)
    )

    fig.update_layout(
        title=f"{holding_period}-Day Portfolio Return Distribution — {selected_model}",
        template="plotly_white",
        xaxis_title=f"{holding_period}-Day Portfolio Return",
        yaxis_title="Probability Density",
        xaxis_tickformat=".1%",
        hovermode="x unified",
        height=500,
        showlegend=False
    )

    return fig




def plot_var_es_comparison(var_comparison):
    """Plot VaR and Expected Shortfall across risk estimation methods."""

    import plotly.express as px

    comparison_data = var_comparison.reset_index()
    comparison_data = comparison_data.rename(columns={"index": "Method"})

    comparison_data = comparison_data.melt(
        id_vars="Method",
        value_vars=["VaR ($)", "ES ($)"],
        var_name="Risk Measure",
        value_name="Amount"
    )

    fig = px.bar(
        comparison_data,
        x="Method",
        y="Amount",
        color="Risk Measure",
        barmode="group",
        text="Amount",
        color_discrete_map={
            "VaR ($)": "#636EFA",
            "ES ($)": "#EF553B"
        },
        labels={
            "Amount": "Potential Loss ($)",
            "Method": "Risk Estimation Method"
        }
    )

    fig.update_traces(
        texttemplate="$%{text:,.2f}",
        textposition="outside",
        cliponaxis=False
    )

    fig.update_layout(
        template="plotly_white",
        height=450,
        yaxis_title="Potential Loss ($)",
        xaxis_title=None,
        legend_title=None,
        margin=dict(t=40, b=40, l=40, r=40),
        bargap=0.3,
        bargroupgap=0.1
    )

    return fig


def calculate_var_sensitivity(
    portfolio_returns,
    asset_returns,
    weights,
    investment_value,
    holding_period=1,
    window=None,
    n_simulations=10000
):
    """Calculate VaR across different confidence levels."""

    confidence_levels = [0.90, 0.95, 0.99]

    results = []

    for confidence in confidence_levels:

        comparison = calculate_var_comparison(
            portfolio_returns=portfolio_returns,
            asset_returns=asset_returns,
            weights=weights,
            investment_value=investment_value,
            confidence_level=confidence,
            holding_period=holding_period,
            window=window,
            n_simulations=n_simulations
        )

        for method in comparison.index:

            results.append({
                "Confidence Level": confidence,
                "Method": method,
                "VaR ($)": comparison.loc[method, "VaR ($)"]
            })

    return pd.DataFrame(results)




def plot_var_sensitivity(sensitivity_data):
    """Plot VaR sensitivity across confidence levels."""

    import plotly.express as px

    sensitivity_data = sensitivity_data.copy()

    sensitivity_data["Confidence Level"] = (
        sensitivity_data["Confidence Level"].map(
            lambda x: f"{x:.0%}"
        )
    )

    fig = px.line(
        sensitivity_data,
        x="Confidence Level",
        y="VaR ($)",
        color="Method",
        markers=True,
        color_discrete_map={
            "Historical": "#EF553B",
            "Parametric": "#FFA15A",
            "Monte Carlo": "#00CC96"
        },
        category_orders={
            "Confidence Level": ["90%", "95%", "99%"]
        }
    )

    fig.update_traces(
        line=dict(width=2.5),
        marker=dict(size=9),
        hovertemplate=(
            "Method: %{fullData.name}<br>"
            "Confidence Level: %{x}<br>"
            "VaR: $%{y:,.2f}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        template="plotly_white",
        title="VaR Sensitivity to Confidence Level",
        xaxis_title="Confidence Level",
        yaxis_title="Value at Risk ($)",
        height=450,
        hovermode="x unified",
        legend_title="Method"
    )

    return fig

