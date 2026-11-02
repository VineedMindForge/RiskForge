"""Portfolio return, value, performance and chart calculations."""

import numpy as np
import pandas as pd
import plotly.express as px


TRADING_DAYS = 252


def calculate_portfolio_returns(prices, weights, rebalance_frequency="Daily"):
    """Calculate daily portfolio returns and the weights used on each day."""

    weights = pd.Series(weights, dtype=float).reindex(prices.columns)

    if weights.isna().any():
        raise ValueError("Provide one weight for every selected asset.")
    if (weights < 0).any():
        raise ValueError("This portfolio calculation supports long-only weights.")
    if not np.isclose(weights.sum(), 1):
        raise ValueError("Portfolio weights must add up to 1.")
    if prices.isna().any().any():
        raise ValueError("Selected assets must have prices for the same dates. Remove incomplete dates first.")
    if (prices <= 0).any().any():
        raise ValueError("Prices must be greater than zero.")

    asset_returns = prices.pct_change(fill_method=None).dropna()
    portfolio_returns = pd.Series(index=asset_returns.index, dtype=float, name="Portfolio Return")
    allocation_drift = pd.DataFrame(index=asset_returns.index, columns=prices.columns, dtype=float)

    if rebalance_frequency == "Daily":
        # Restore the target weights before every day's return.
        portfolio_returns = asset_returns @ weights
        allocation_drift.loc[:, :] = weights.values

    elif rebalance_frequency == "Buy and Hold":
        # The value of each holding changes with its market price.
        asset_values = (prices / prices.iloc[0]).mul(weights, axis=1)
        portfolio_index = asset_values.sum(axis=1)
        portfolio_returns = portfolio_index.pct_change(fill_method=None).dropna()

        # Show the weights held at the start of each return period.
        previous_values = asset_values.shift(1).iloc[1:]
        allocation_drift = previous_values.div(previous_values.sum(axis=1), axis=0)

    elif rebalance_frequency == "Monthly":
        current_weights = weights.copy()
        previous_month = None

        for date, daily_asset_returns in asset_returns.iterrows():
            current_month = date.to_period("M")

            # Restore target weights on the first available trading day of a new month.
            if previous_month is not None and current_month != previous_month:
                current_weights = weights.copy()

            allocation_drift.loc[date] = current_weights
            portfolio_returns.loc[date] = daily_asset_returns @ current_weights

            # Let the weights drift after the daily price movement.
            end_values = current_weights * (1 + daily_asset_returns)
            current_weights = end_values / end_values.sum()
            previous_month = current_month

    else:
        raise ValueError("Choose Daily, Monthly or Buy and Hold rebalancing.")

    return portfolio_returns, allocation_drift


def calculate_portfolio_value(initial_investment, portfolio_returns, initial_date):
    """Compound daily returns and include the initial investment date and value."""

    portfolio_values = initial_investment * (1 + portfolio_returns).cumprod()
    initial_value = pd.Series([initial_investment], index=[pd.Timestamp(initial_date)])
    portfolio_values = pd.concat([initial_value, portfolio_values])
    portfolio_values.name = "Portfolio Value"
    return portfolio_values


def calculate_returns_metrics(portfolio_value):
    """Return the cumulative return and CAGR of a portfolio value series."""

    initial_value = portfolio_value.iloc[0]
    final_value = portfolio_value.iloc[-1]
    years = (len(portfolio_value) - 1) / TRADING_DAYS

    total_return = final_value / initial_value - 1
    cagr = (final_value / initial_value) ** (1 / years) - 1
    return total_return, cagr


def calculate_volatility_metrics(portfolio_returns):
    """Return daily and annualised sample volatility."""

    daily_volatility = portfolio_returns.std()
    annual_volatility = daily_volatility * np.sqrt(TRADING_DAYS)
    return daily_volatility, annual_volatility


def calculate_portfolio_metrics(portfolio_value, risk_free_rate=0.02):
    """Calculate total return, CAGR, volatility and Sharpe ratio."""

    portfolio_returns = portfolio_value.pct_change(fill_method=None).dropna()
    total_return, cagr = calculate_returns_metrics(portfolio_value)
    daily_volatility, annual_volatility = calculate_volatility_metrics(portfolio_returns)

    annual_return = portfolio_returns.mean() * TRADING_DAYS
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else np.nan

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Daily Volatility": daily_volatility,
        "Annualized Volatility": annual_volatility,
        "Sharpe Ratio": sharpe_ratio
    }


def plot_portfolio_vs_assets(prices, portfolio_value):
    """Plot portfolio value against each asset invested in on its own."""

    initial_investment = portfolio_value.iloc[0]
    asset_values = prices.div(prices.iloc[0]).mul(initial_investment)
    comparison = asset_values.join(portfolio_value.rename("Portfolio"), how="inner")

    fig = px.line(comparison, x=comparison.index, y=comparison.columns,
                  title="Portfolio Performance vs Individual Assets")
    for trace in fig.data:
        if trace.name == "Portfolio":
            trace.line.width = 3
        else:
            trace.opacity = 0.35
    fig.update_layout(template="plotly_white", hovermode="x unified", yaxis_title="Portfolio Value")
    return fig


def plot_allocation_over_time(allocation_drift):
    """Plot how the portfolio weights change over time."""

    fig = px.area(allocation_drift * 100, x=allocation_drift.index, y=allocation_drift.columns,
                  title="Portfolio Allocation Over Time")
    fig.update_layout(template="plotly_white", hovermode="x unified",
                      yaxis_title="Allocation (%)", yaxis_range=[0, 100])
    return fig


def plot_current_allocation(allocation_drift):
    """Plot the latest portfolio weights as a donut chart."""

    current_weights = allocation_drift.iloc[-1].dropna()
    fig = px.pie(names=current_weights.index, values=current_weights.values,
                 hole=0.5, title="Latest Portfolio Allocation")
    fig.update_traces(textinfo="label+percent")
    fig.update_layout(template="plotly_white")
    return fig
