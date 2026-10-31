"""Portfolio construction, return simulation, performance metrics and charts.

This module contains the reusable calculations used by the Streamlit
Portfolio Construction page. It intentionally contains no Streamlit widgets
or application layout code: callers supply user selections and display the
returned Pandas objects / Plotly figures.

Main workflow
-------------
1. Select a complete, positive-price subset for the investment period.
2. Convert target allocations into a ticker-indexed weights Series.
3. Call :func:`calculate_portfolio_returns` to obtain daily returns and
   beginning-of-day allocation weights under the chosen rebalancing policy.
4. Call :func:`calculate_portfolio_value` to build a value series that
   includes the initial investment date and subsequent end-of-day values.
5. Call :func:`calculate_portfolio_metrics` for total return, CAGR, volatility
   and Sharpe ratio.
6. Use the plotting helpers to create reusable Plotly figures.

Conventions
-----------
* Prices are assumed to be adjusted daily closing prices.
* Returns are simple percentage returns, not log returns.
* Portfolio weights are long-only, sum to 1, and are aligned by ticker.
* Rebalancing is assumed to occur without transaction costs or taxes.
* The first investment-period price is the initial valuation date. The first
  calculated daily return is from that date to the next available observation.
* The allocation DataFrame records beginning-of-day weights, i.e. the weights
  used to calculate that day's portfolio return.
"""

from __future__ import annotations

from typing import Mapping, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

WeightInput = Union[pd.Series, Mapping[str, float]]
TRADING_DAYS_PER_YEAR = 252


def _validate_prices_and_weights(prices: pd.DataFrame, weights: WeightInput) -> pd.Series:
    """Validate portfolio inputs and return weights aligned to price columns.

    This private helper is shared by portfolio calculations. It requires a
    sorted, unique DatetimeIndex, finite positive prices, identical ticker
    coverage and non-negative weights summing to one.
    """
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty pandas DataFrame.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex.")
    if prices.index.has_duplicates:
        raise ValueError("prices contains duplicate dates. Resolve them before portfolio simulation.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices must be sorted in ascending date order.")
    if prices.isna().any().any():
        raise ValueError("prices contains missing values. Align/resolve the selected assets before simulation.")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("prices must contain only finite, strictly positive values.")

    aligned = pd.Series(weights, dtype=float)
    missing_tickers = prices.columns.difference(aligned.index)
    extra_tickers = aligned.index.difference(prices.columns)
    if len(missing_tickers) or len(extra_tickers):
        raise ValueError(
            f"Weights must match price columns. Missing weights: {list(missing_tickers)}; "
            f"unexpected weights: {list(extra_tickers)}."
        )
    aligned = aligned.reindex(prices.columns)
    if aligned.isna().any() or not np.isfinite(aligned.to_numpy()).all():
        raise ValueError("weights must be finite numeric values.")
    if (aligned < 0).any():
        raise ValueError("This implementation supports long-only portfolios; weights cannot be negative.")
    if not np.isclose(aligned.sum(), 1.0, atol=1e-8):
        raise ValueError(f"Portfolio weights must sum to 1. Current sum: {aligned.sum():.8f}.")
    return aligned


def _normalise_rebalancing_frequency(rebalance_frequency: str) -> str:
    """Normalize supported rebalancing labels and a legacy notebook alias."""
    key = str(rebalance_frequency).strip().lower().replace("_", " ")
    aliases = {
        "daily": "Daily",
        "monthly": "Monthly",
        "buy and hold": "Buy and Hold",
        "buy&hold": "Buy and Hold",
        "buyhold": "Buy and Hold",
        "buynhold": "Buy and Hold",  # legacy spelling in the notebook
    }
    if key not in aliases:
        raise ValueError("rebalance_frequency must be 'Daily', 'Monthly', or 'Buy and Hold'.")
    return aliases[key]


def calculate_portfolio_returns(
    prices: pd.DataFrame,
    weights: WeightInput,
    rebalance_frequency: str = "Daily",
) -> tuple[pd.Series, pd.DataFrame]:
    """Calculate daily portfolio returns and beginning-of-day asset weights.

    Parameters
    ----------
    prices:
        Positive adjusted-price DataFrame indexed by trading date. All
        selected assets must have complete observations on the same dates.
    weights:
        Target allocation, supplied as a Pandas Series or mapping keyed by
        ticker. Values must be non-negative and sum to 1.
    rebalance_frequency:
        ``"Daily"``, ``"Monthly"`` or ``"Buy and Hold"``. The old notebook
        spelling ``"BuynHold"`` is accepted for compatibility.

    Returns
    -------
    tuple[pandas.Series, pandas.DataFrame]
        ``portfolio_returns``: simple daily portfolio return series indexed
        by the date on which each return is realised;
        ``allocation_drift``: asset weights at the beginning of each return
        interval. Rows are aligned exactly to ``portfolio_returns``.

    Methodology
    -----------
    Daily
        Apply the target weights to every day's asset returns. This represents
        restoring target allocations at the beginning of every trading day.
    Buy and Hold
        Hold the initial asset quantities. Asset weights drift with relative
        price performance. Portfolio value is calculated from each asset's
        growth relative to its initial price.
    Monthly
        Allow weights to drift during the month. Restore target weights on
        the first available trading observation of each new calendar month,
        before that day's return is applied.

    Notes
    -----
    The function does not model transaction costs, taxes, cash flows, short
    selling, fractional-share constraints or market impact. It fails loudly
    on missing or invalid prices rather than silently filling observations.
    """
    target_weights = _validate_prices_and_weights(prices, weights)
    frequency = _normalise_rebalancing_frequency(rebalance_frequency)
    asset_returns = prices.pct_change(fill_method=None).iloc[1:]

    portfolio_returns = pd.Series(index=asset_returns.index, dtype=float, name="Portfolio Return")
    allocation_drift = pd.DataFrame(
        index=asset_returns.index,
        columns=prices.columns,
        dtype=float,
    )

    if frequency == "Daily":
        portfolio_returns.loc[:] = asset_returns @ target_weights
        allocation_drift.loc[:, :] = target_weights.to_numpy()

    elif frequency == "Buy and Hold":
        # Value of each holding relative to the initial total portfolio value.
        relative_asset_values = (prices / prices.iloc[0]).mul(target_weights, axis=1)
        portfolio_value_index = relative_asset_values.sum(axis=1)
        portfolio_returns.loc[:] = portfolio_value_index.pct_change(fill_method=None).iloc[1:].to_numpy()

        # Beginning-of-day weights are based on the previous close.
        previous_values = relative_asset_values.shift(1).iloc[1:]
        allocation_drift.loc[:, :] = previous_values.div(previous_values.sum(axis=1), axis=0).to_numpy()

    else:  # Monthly
        current_weights = target_weights.copy()
        previous_month = None

        for date, daily_asset_returns in asset_returns.iterrows():
            current_month = date.to_period("M")

            # Rebalance before applying the first return observed in a new month.
            if previous_month is not None and current_month != previous_month:
                current_weights = target_weights.copy()

            allocation_drift.loc[date] = current_weights
            portfolio_returns.loc[date] = float(daily_asset_returns @ current_weights)

            # Let the asset weights drift after today's price movement.
            end_of_day_values = current_weights * (1.0 + daily_asset_returns)
            current_weights = end_of_day_values / end_of_day_values.sum()
            previous_month = current_month

    return portfolio_returns, allocation_drift


def calculate_portfolio_value(
    portfolio_returns: pd.Series,
    initial_investment: float = 100_000,
    initial_date: Union[str, pd.Timestamp, None] = None,
) -> pd.Series:
    """Compound daily portfolio returns into a value series including capital at inception.

    Parameters
    ----------
    portfolio_returns:
        Daily simple returns indexed by the dates on which those returns are
        realised. The first value is the return from the initial valuation date
        to the first return date.
    initial_investment:
        Positive starting portfolio value, in the chosen reporting currency.
    initial_date:
        Date of the initial investment / first price observation. Pass
        ``prices.index[0]`` from the investment-period price DataFrame. This
        date is prepended to the returned value series.

    Returns
    -------
    pandas.Series
        Portfolio values with the initial investment at ``initial_date`` and
        end-of-day portfolio values on each return date.

    Notes
    -----
    Explicitly supplying ``initial_date`` avoids assigning the initial capital
    to the date of the first realised return. This keeps portfolio value,
    total-return calculations and constituent-asset comparison charts aligned.
    """
    if not isinstance(portfolio_returns, pd.Series) or portfolio_returns.empty:
        raise ValueError("portfolio_returns must be a non-empty Pandas Series.")
    if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
        raise TypeError("portfolio_returns must have a DatetimeIndex.")
    if portfolio_returns.index.has_duplicates or not portfolio_returns.index.is_monotonic_increasing:
        raise ValueError("portfolio_returns must have unique, ascending dates.")
    if not np.isfinite(portfolio_returns.to_numpy(dtype=float)).all():
        raise ValueError("portfolio_returns contains non-finite values.")
    if (portfolio_returns <= -1).any():
        raise ValueError("A simple portfolio return cannot be less than or equal to -100%.")
    if not np.isfinite(initial_investment) or initial_investment <= 0:
        raise ValueError("initial_investment must be a finite positive number.")
    if initial_date is None:
        raise ValueError("initial_date is required. Pass the first date in the investment price series.")

    initial_date = pd.Timestamp(initial_date)
    if initial_date >= portfolio_returns.index[0]:
        raise ValueError("initial_date must be earlier than the first portfolio return date.")

    compounded_values = initial_investment * (1.0 + portfolio_returns).cumprod()
    value_series = pd.concat(
        [
            pd.Series([float(initial_investment)], index=pd.DatetimeIndex([initial_date])),
            compounded_values,
        ]
    )
    value_series.name = "Portfolio Value"
    return value_series


def calculate_returns_metrics(portfolio_value: pd.Series) -> tuple[float, float]:
    """Calculate cumulative total return and CAGR from a portfolio value series.

    Parameters
    ----------
    portfolio_value:
        Positive portfolio value Series including the initial investment
        observation as its first row.

    Returns
    -------
    tuple[float, float]
        ``total_return``: final value divided by initial value minus one;
        ``cagr``: compounded annual growth rate using 252 trading observations
        per year and the number of return intervals in the series.
    """
    if not isinstance(portfolio_value, pd.Series) or len(portfolio_value) < 2:
        raise ValueError("portfolio_value must contain an initial value and at least one subsequent value.")
    if (portfolio_value <= 0).any() or not np.isfinite(portfolio_value.to_numpy(dtype=float)).all():
        raise ValueError("portfolio_value must contain finite, positive values.")

    initial_value = float(portfolio_value.iloc[0])
    final_value = float(portfolio_value.iloc[-1])
    number_of_years = (len(portfolio_value) - 1) / TRADING_DAYS_PER_YEAR
    total_return = final_value / initial_value - 1.0
    cagr = (final_value / initial_value) ** (1.0 / number_of_years) - 1.0
    return float(total_return), float(cagr)


def calculate_volatility_metrics(
    portfolio_returns: pd.Series,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> tuple[float, float]:
    """Calculate daily and annualised sample volatility of portfolio returns.

    Parameters
    ----------
    portfolio_returns:
        Daily simple portfolio returns.
    annualization_factor:
        Number of trading observations per year; 252 is the conventional
        assumption for daily US-market data.

    Returns
    -------
    tuple[float, float]
        Daily standard deviation and annualised standard deviation.
    """
    clean_returns = portfolio_returns.dropna()
    if len(clean_returns) < 2:
        raise ValueError("At least two non-missing returns are required to estimate volatility.")
    daily_volatility = float(clean_returns.std(ddof=1))
    annualized_volatility = daily_volatility * np.sqrt(annualization_factor)
    return daily_volatility, float(annualized_volatility)


def calculate_portfolio_metrics(
    portfolio_value: pd.Series,
    risk_free_rate: float = 0.02,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Calculate headline portfolio performance and risk metrics.

    Parameters
    ----------
    portfolio_value:
        Portfolio value Series including the initial investment point.
    risk_free_rate:
        Annual risk-free rate as a decimal; for example, 0.04 represents 4%.
    annualization_factor:
        Trading observations per year, normally 252 for daily data.

    Returns
    -------
    dict[str, float]
        Total Return, CAGR, Daily Volatility, Annualized Volatility and
        Sharpe Ratio.

    Notes
    -----
    The Sharpe numerator uses the arithmetic annualised mean daily return
    (mean daily return multiplied by 252) less the annual risk-free rate.
    The denominator is annualised daily-return volatility. This is the
    conventional arithmetic annualisation approximation used in this project.
    A zero-volatility portfolio has an undefined Sharpe ratio, returned as NaN.
    """
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be a finite annual decimal rate.")

    portfolio_returns = portfolio_value.pct_change(fill_method=None).dropna()
    total_return, cagr = calculate_returns_metrics(portfolio_value)
    daily_volatility, annualized_volatility = calculate_volatility_metrics(
        portfolio_returns, annualization_factor
    )

    annualized_arithmetic_return = float(portfolio_returns.mean() * annualization_factor)
    sharpe_ratio = (
        (annualized_arithmetic_return - risk_free_rate) / annualized_volatility
        if annualized_volatility > 0
        else np.nan
    )

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Daily Volatility": daily_volatility,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": float(sharpe_ratio),
    }


def plot_portfolio_vs_assets(
    prices: pd.DataFrame,
    portfolio_value: pd.Series,
    title: str = "Portfolio Performance vs. Individual Assets",
) -> go.Figure:
    """Create a Plotly chart comparing portfolio value with standalone assets.

    Each asset is normalised to the same initial investment as the portfolio.
    These asset lines are standalone benchmarks, not each asset's contribution
    to the constructed portfolio.

    Parameters
    ----------
    prices:
        Investment-period price DataFrame, including the initial date.
    portfolio_value:
        Portfolio value Series including the initial date.
    title:
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive multi-line Plotly figure.
    """
    if prices.empty or portfolio_value.empty:
        raise ValueError("prices and portfolio_value must not be empty.")
    initial_investment = float(portfolio_value.iloc[0])
    normalized_assets = prices.div(prices.iloc[0]).mul(initial_investment)
    comparison = normalized_assets.join(portfolio_value.rename("Portfolio"), how="inner")
    fig = px.line(
        comparison,
        x=comparison.index,
        y=comparison.columns,
        title=title,
        labels={"x": "Date", "value": "Value", "variable": "Investment"},
    )
    for trace in fig.data:
        if trace.name == "Portfolio":
            trace.line.width = 3
            trace.opacity = 1.0
        else:
            trace.opacity = 0.35
    fig.update_layout(template="plotly_white", hovermode="x unified", yaxis_title="Value")
    return fig


def plot_allocation_over_time(
    allocation_drift: pd.DataFrame,
    title: str = "Portfolio Allocation Over Time",
) -> go.Figure:
    """Create a 100% stacked area chart of beginning-of-day portfolio weights.

    Parameters
    ----------
    allocation_drift:
        DataFrame returned by :func:`calculate_portfolio_returns`; rows are
        return dates and columns are asset weights.
    title:
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive stacked area chart. Input weights are converted to percent.
    """
    if allocation_drift.empty:
        raise ValueError("allocation_drift must not be empty.")
    fig = px.area(
        allocation_drift * 100,
        x=allocation_drift.index,
        y=allocation_drift.columns,
        title=title,
        labels={"x": "Date", "value": "Allocation (%)", "variable": "Asset"},
    )
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        yaxis_title="Allocation (%)",
        yaxis_range=[0, 100],
        legend_title="Asset",
    )
    return fig


def plot_current_allocation(
    allocation_drift: pd.DataFrame,
    title: str = "Latest Portfolio Allocation",
) -> go.Figure:
    """Create a donut chart of the latest available portfolio weights.

    Parameters
    ----------
    allocation_drift:
        DataFrame of beginning-of-day weights returned by
        :func:`calculate_portfolio_returns`.
    title:
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly donut chart.
    """
    if allocation_drift.empty:
        raise ValueError("allocation_drift must not be empty.")
    current_weights = allocation_drift.iloc[-1].dropna()
    fig = px.pie(
        names=current_weights.index,
        values=current_weights.values,
        title=title,
        hole=0.5,
    )
    fig.update_traces(textinfo="label+percent", hovertemplate="%{label}: %{value:.2%}<extra></extra>")
    fig.update_layout(template="plotly_white")
    return fig
