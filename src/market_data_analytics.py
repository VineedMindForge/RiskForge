"""Asset-level market data analytics and Plotly visualisation functions.

This module analyses individual assets independently of portfolio weights and
portfolio rebalancing. It is intended for the Streamlit Market Data & Analytics
page, but all functions are also usable from a Jupyter notebook.

Main workflow
-------------
>>> daily_returns = calculate_asset_returns(prices)
>>> asset_stats = calculate_asset_statistics(prices)
>>> return_stats = calculate_return_statistics(daily_returns)
>>> volatility_by_asset, full_sample_vol = vol_analysis(daily_returns)
>>> corr_matrix = calculate_correlation_matrix(daily_returns)
>>> normality = calculate_normality_test(daily_returns)

Plotting functions return Plotly figures. They do not call ``fig.show()`` or
Streamlit's ``st.plotly_chart()``. The notebook can call ``fig.show()``; a
Streamlit page can call ``st.plotly_chart(fig, use_container_width=True)``.

Conventions and interpretation
------------------------------
* Inputs are adjusted daily closing prices and simple daily returns.
* The standard annualisation factor is 252 trading days.
* Volatility outputs are annualised decimals (e.g. 0.20 means 20%).
* EWMA volatility uses a zero-mean RiskMetrics-style squared-return estimate.
* GARCH volatility uses a zero-mean GARCH(1,1) model with normal innovations.
* Jarque-Bera results test the null hypothesis that returns are normally
  distributed. A p-value above the significance level means normality is not
  rejected; it does not prove that returns are normal.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import jarque_bera, norm

TRADING_DAYS_PER_YEAR = 252


def _validate_returns(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a numeric daily-returns DataFrame (private helper)."""
    if not isinstance(daily_returns, pd.DataFrame) or daily_returns.empty:
        raise ValueError("daily_returns must be a non-empty pandas DataFrame.")
    if not isinstance(daily_returns.index, pd.DatetimeIndex):
        raise TypeError("daily_returns must have a DatetimeIndex.")
    if daily_returns.index.has_duplicates:
        raise ValueError("daily_returns contains duplicate dates.")
    if not daily_returns.index.is_monotonic_increasing:
        raise ValueError("daily_returns must be sorted in ascending date order.")
    return daily_returns.apply(pd.to_numeric, errors="coerce")


def calculate_asset_statistics(prices: pd.DataFrame) -> pd.DataFrame:
    """Summarise date coverage, price completeness and latest valid price per asset.

    Parameters
    ----------
    prices:
        Adjusted-price DataFrame indexed by date, with one column per ticker.

    Returns
    -------
    pandas.DataFrame
        One row per asset, indexed by ticker. Columns include first/last valid
        dates, observation count, missing count/percentage, completeness,
        invalid-price count, latest valid positive price/date and a warning flag.

    Notes
    -----
    Missing values and invalid values are counted separately. Invalid means a
    non-missing price that is non-finite or less than/equal to zero. The report
    does not clean the input prices.
    """
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty pandas DataFrame.")

    records = []
    total_rows = len(prices)

    for ticker in prices.columns:
        asset_prices = pd.to_numeric(prices[ticker], errors="coerce")
        missing = int(asset_prices.isna().sum())
        observations = int(asset_prices.notna().sum())
        missing_percentage = missing / total_rows * 100

        invalid_mask = asset_prices.notna() & (
            (asset_prices <= 0) | ~np.isfinite(asset_prices)
        )
        invalid_count = int(invalid_mask.sum())
        valid_prices = asset_prices[asset_prices.notna() & ~invalid_mask]

        records.append(
            {
                "ticker": ticker,
                "first_date": valid_prices.index.min() if not valid_prices.empty else pd.NaT,
                "last_date": valid_prices.index.max() if not valid_prices.empty else pd.NaT,
                "missing_observations": missing,
                "missing_percentage": missing_percentage,
                "number_of_observations": observations,
                "data_completeness_percentage": 100 - missing_percentage,
                "latest_valid_price": valid_prices.iloc[-1] if not valid_prices.empty else np.nan,
                "latest_valid_price_date": valid_prices.index[-1] if not valid_prices.empty else pd.NaT,
                "invalid_prices": invalid_count,
                "data_validation_warning": bool(missing > 0 or invalid_count > 0),
            }
        )

    return pd.DataFrame(records).set_index("ticker")


def calculate_asset_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple daily percentage returns for each asset.

    Parameters
    ----------
    prices:
        Adjusted-price DataFrame indexed by date.

    Returns
    -------
    pandas.DataFrame
        Daily simple returns, retaining the first-row NaNs and any missing
        observations. Keeping those NaNs visible avoids silently changing the
        return series through implicit forward-filling. Downstream estimators
        should explicitly drop missing observations per asset as required.
    """
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty pandas DataFrame.")
    return prices.pct_change(fill_method=None)


def calculate_return_statistics(
    daily_returns: pd.DataFrame,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Calculate descriptive return statistics for each individual asset.

    Parameters
    ----------
    daily_returns:
        DataFrame of daily simple returns, one column per asset.
    annualization_factor:
        Number of trading observations per year, normally 252.

    Returns
    -------
    pandas.DataFrame
        Per-asset daily mean, arithmetic annualised mean return, daily and
        annualised volatility, skewness, excess kurtosis, and best/worst daily
        return. ``annualized_return`` is mean daily return multiplied by the
        annualisation factor; it is not CAGR.
    """
    daily_returns = _validate_returns(daily_returns)
    records = []

    for ticker in daily_returns.columns:
        asset_returns = daily_returns[ticker].replace([np.inf, -np.inf], np.nan).dropna()
        if len(asset_returns) < 2:
            daily_mean = annualized_return = daily_vol = annualized_vol = np.nan
            skewness = excess_kurtosis = best_return = worst_return = np.nan
        else:
            daily_mean = asset_returns.mean()
            annualized_return = daily_mean * annualization_factor
            daily_vol = asset_returns.std(ddof=1)
            annualized_vol = daily_vol * np.sqrt(annualization_factor)
            skewness = asset_returns.skew()
            excess_kurtosis = asset_returns.kurt()  # Pandas uses Fisher/excess kurtosis.
            best_return = asset_returns.max()
            worst_return = asset_returns.min()

        records.append(
            {
                "ticker": ticker,
                "daily_mean_return": daily_mean,
                "annualized_arithmetic_return": annualized_return,
                "daily_volatility": daily_vol,
                "annualized_volatility": annualized_vol,
                "skewness": skewness,
                "excess_kurtosis": excess_kurtosis,
                "best_daily_return": best_return,
                "worst_daily_return": worst_return,
            }
        )

    return pd.DataFrame(records).set_index("ticker")


def calculate_rolling_vol(
    asset_returns: pd.Series,
    window: int = 30,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Calculate annualised rolling sample volatility for one asset.

    The estimate at date t uses the current observation and the preceding
    ``window - 1`` observations. Results remain NaN until a full window exists.
    """
    if window < 2:
        raise ValueError("window must be at least 2 observations.")
    clean = asset_returns.replace([np.inf, -np.inf], np.nan)
    return clean.rolling(window=window, min_periods=window).std(ddof=1) * np.sqrt(annualization_factor)


def calculate_ewma_vol(
    asset_returns: pd.Series,
    lambda_: float = 0.94,
    window: int = 30,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Calculate annualised EWMA volatility for one asset.

    Parameters
    ----------
    asset_returns:
        Daily simple returns for one asset.
    lambda_:
        EWMA decay factor in (0, 1). Larger values give older observations
        more persistent influence. 0.94 is the commonly used daily RiskMetrics
        starting value.
    window:
        Minimum number of valid observations before an estimate is emitted.
    annualization_factor:
        Trading observations per year, normally 252.

    Returns
    -------
    pandas.Series
        Annualised EWMA volatility as a decimal.

    Notes
    -----
    The variance estimate is the exponentially weighted mean of squared
    returns, equivalent to assuming a zero conditional mean. This is a
    deliberate RiskMetrics-style convention; it is not a demeaned EWMA model.
    """
    if not 0 < lambda_ < 1:
        raise ValueError("lambda_ must be strictly between 0 and 1.")
    if window < 1:
        raise ValueError("window must be at least 1.")
    clean = asset_returns.replace([np.inf, -np.inf], np.nan)
    ewma_variance = clean.pow(2).ewm(
        alpha=1 - lambda_,
        min_periods=window,
        adjust=False,
        ignore_na=True,
    ).mean()
    return np.sqrt(ewma_variance) * np.sqrt(annualization_factor)


def calculate_garch_vol(
    asset_returns: pd.Series,
    p: int = 1,
    q: int = 1,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Fit a zero-mean GARCH(p,q) model and return annualised conditional volatility.

    Parameters
    ----------
    asset_returns:
        Daily simple returns for one asset, expressed as decimals.
    p, q:
        ARCH and GARCH orders. Defaults (1, 1) give GARCH(1,1).
    annualization_factor:
        Trading observations per year, normally 252.

    Returns
    -------
    pandas.Series
        Annualised conditional volatility as a decimal, indexed by the valid
        input return dates.

    Notes
    -----
    Returns are multiplied by 100 before fitting for numerical scaling, then
    conditional standard deviations are divided by 100 and annualised.
    The ``arch`` package is imported inside this function so the other market
    analytics remain usable if the optional GARCH dependency is not installed.
    """
    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError("GARCH estimation requires the 'arch' package. Install it with: pip install arch") from exc

    clean = asset_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < max(50, p + q + 10):
        raise ValueError("Insufficient observations for GARCH estimation; provide at least 50 daily returns.")
    if p < 1 or q < 1:
        raise ValueError("GARCH orders p and q must both be positive integers.")

    model = arch_model(clean * 100, mean="Zero", vol="GARCH", p=p, q=q, dist="normal")
    fitted = model.fit(disp="off")
    conditional_volatility = fitted.conditional_volatility / 100
    return conditional_volatility * np.sqrt(annualization_factor)


def vol_analysis(
    daily_returns: pd.DataFrame,
    window: int = 30,
    lambda_: float = 0.94,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Calculate rolling, EWMA and GARCH volatility for every asset.

    Parameters
    ----------
    daily_returns:
        Daily simple-return DataFrame, one column per ticker.
    window:
        Rolling-volatility window and EWMA minimum initialisation observations.
    lambda_:
        EWMA decay factor.
    annualization_factor:
        Trading observations per year.

    Returns
    -------
    tuple[dict[str, pandas.DataFrame], pandas.Series]
        ``vol_stats`` maps each ticker to a DataFrame with ``rolling_vol``,
        ``ewma_vol`` and ``garch_vol`` columns; ``annual_asset_vol`` contains
        each asset's full-sample annualised historical standard deviation.

    Notes
    -----
    GARCH is fitted separately for each asset and may be computationally
    expensive. If a model fails to converge for one asset, the exception is
    surfaced rather than silently replacing the estimate.
    """
    daily_returns = _validate_returns(daily_returns)
    vol_stats: dict[str, pd.DataFrame] = {}

    for ticker in daily_returns.columns:
        asset_returns = daily_returns[ticker]
        rolling_vol = calculate_rolling_vol(asset_returns, window, annualization_factor)
        ewma_vol = calculate_ewma_vol(asset_returns, lambda_, window, annualization_factor)
        garch_vol = calculate_garch_vol(asset_returns, annualization_factor=annualization_factor)
        vol_stats[ticker] = pd.concat(
            [
                rolling_vol.rename("rolling_vol"),
                ewma_vol.rename("ewma_vol"),
                garch_vol.rename("garch_vol"),
            ],
            axis=1,
        )

    annual_asset_vol = daily_returns.std(ddof=1) * np.sqrt(annualization_factor)
    annual_asset_vol.name = "annualized_historical_volatility"
    return vol_stats, annual_asset_vol


def calculate_correlation_matrix(
    daily_returns: pd.DataFrame,
    window: Optional[int] = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Calculate the asset correlation matrix over full history or a recent window.

    Parameters
    ----------
    daily_returns:
        Daily return DataFrame, one column per asset.
    window:
        If supplied, use only the most recent ``window`` rows. If None, use
        the full supplied history.
    method:
        Pandas correlation method: ``"pearson"``, ``"spearman"`` or ``"kendall"``.

    Returns
    -------
    pandas.DataFrame
        Symmetric correlation matrix. Pandas uses pairwise complete observations
        when missing values are present.
    """
    daily_returns = _validate_returns(daily_returns)
    if window is not None:
        if window < 2:
            raise ValueError("window must be at least 2 observations.")
        daily_returns = daily_returns.tail(window)
    return daily_returns.corr(method=method)


def calculate_rolling_correlation(
    daily_returns: pd.DataFrame,
    asset1: str,
    asset2: str,
    window: int = 60,
) -> pd.Series:
    """Calculate rolling Pearson correlation for a selected pair of assets.

    Parameters
    ----------
    daily_returns:
        Daily return DataFrame containing both selected assets.
    asset1, asset2:
        Column names identifying the two assets. They must be different.
    window:
        Number of paired daily observations in each rolling window.

    Returns
    -------
    pandas.Series
        Rolling correlation indexed by date. The first ``window - 1`` values
        are NaN because a full observation window is not yet available.
    """
    daily_returns = _validate_returns(daily_returns)
    if asset1 not in daily_returns.columns or asset2 not in daily_returns.columns:
        raise KeyError("Both selected assets must exist in daily_returns columns.")
    if asset1 == asset2:
        raise ValueError("Select two different assets for rolling correlation.")
    if window < 2:
        raise ValueError("window must be at least 2 observations.")
    return daily_returns[asset1].rolling(window).corr(daily_returns[asset2])


def calculate_normality_test(
    asset_returns: pd.DataFrame,
    significance_level: float = 0.05,
) -> pd.DataFrame:
    """Run the Jarque-Bera normality test on each asset's daily returns.

    Parameters
    ----------
    asset_returns:
        Daily return DataFrame, one column per asset.
    significance_level:
        Test significance threshold, normally 0.05.

    Returns
    -------
    pandas.DataFrame
        One row per asset with observation count, JB statistic, p-value and
        ``normality_not_rejected`` Boolean. True means the null of normality
        was not rejected at the chosen significance level; it does not prove
        that the population distribution is normal.
    """
    if not 0 < significance_level < 1:
        raise ValueError("significance_level must be between 0 and 1.")
    asset_returns = _validate_returns(asset_returns)
    records = []

    for ticker in asset_returns.columns:
        clean = asset_returns[ticker].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 3:
            statistic, p_value = np.nan, np.nan
            not_rejected = False
        else:
            result = jarque_bera(clean)
            statistic, p_value = float(result.statistic), float(result.pvalue)
            not_rejected = bool(p_value >= significance_level)

        records.append(
            {
                "ticker": ticker,
                "observations": len(clean),
                "jb_statistic": statistic,
                "p_value": p_value,
                "normality_not_rejected": not_rejected,
            }
        )

    return pd.DataFrame(records).set_index("ticker")


# ---------------------------------------------------------------------------
# Plotly visualisation functions
# ---------------------------------------------------------------------------


def plot_normalized_asset_prices(prices: pd.DataFrame, base_value: float = 100.0) -> go.Figure:
    """Plot all asset prices rebased to a common starting index (default 100).

    This permits comparison of percentage price performance when assets have
    different nominal share prices. It is a standalone-asset comparison, not a
    portfolio simulation.
    """
    if prices.empty:
        raise ValueError("prices must not be empty.")
    normalized = prices.div(prices.iloc[0]).mul(base_value)
    fig = px.line(
        normalized,
        x=normalized.index,
        y=normalized.columns,
        title="Historical Asset Performance (Initial = 100)",
        labels={"x": "Date", "value": "Indexed Price", "variable": "Asset"},
    )
    fig.update_layout(template="plotly_white", hovermode="x unified", yaxis_title=f"Indexed Price (Initial = {base_value:g})")
    return fig


def plot_asset_returns(ticker: str, daily_returns: pd.DataFrame) -> go.Figure:
    """Plot the daily return time series for one selected asset."""
    if ticker not in daily_returns.columns:
        raise KeyError(f"{ticker} is not present in daily_returns.")
    series = daily_returns[ticker].dropna()
    fig = go.Figure(
        go.Scatter(
            x=series.index,
            y=series,
            mode="lines",
            name="Daily Returns",
            line={"color": "#636EFA", "width": 1},
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
    fig.update_layout(
        title=f"{ticker} - Daily Returns",
        template="plotly_white",
        height=400,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Daily Return",
        yaxis_tickformat=".1%",
    )
    return fig


def plot_return_distribution(ticker: str, daily_returns: pd.DataFrame) -> go.Figure:
    """Plot one asset's daily-return histogram, fitted normal curve and mean line.

    The normal curve is a visual reference using the sample mean and standard
    deviation; it is not a fitted heavy-tailed or skewed distribution model.
    """
    if ticker not in daily_returns.columns:
        raise KeyError(f"{ticker} is not present in daily_returns.")
    asset_returns = daily_returns[ticker].replace([np.inf, -np.inf], np.nan).dropna()
    if len(asset_returns) < 2 or asset_returns.std(ddof=1) == 0:
        raise ValueError(f"Not enough variation in {ticker} returns to plot a distribution.")

    mean = float(asset_returns.mean())
    standard_deviation = float(asset_returns.std(ddof=1))
    x_values = np.linspace(asset_returns.min(), asset_returns.max(), 300)
    normal_density = norm.pdf(x_values, mean, standard_deviation)

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=asset_returns,
            histnorm="probability density",
            name="Observed Returns",
            opacity=0.65,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=normal_density,
            mode="lines",
            name="Normal Reference",
            line={"color": "red", "width": 2},
        )
    )
    fig.add_vline(x=mean, line_dash="dash", line_color="black", annotation_text="Mean")
    fig.update_layout(
        title=f"{ticker} - Daily Return Distribution",
        template="plotly_white",
        height=400,
        xaxis_title="Daily Return",
        yaxis_title="Density",
        xaxis_tickformat=".1%",
    )
    return fig


def plot_return_boxplot(daily_returns: pd.DataFrame) -> go.Figure:
    """Compare daily-return distributions and outliers across all assets."""
    fig = go.Figure()
    for ticker in daily_returns.columns:
        fig.add_trace(
            go.Box(
                y=daily_returns[ticker].dropna(),
                name=ticker,
                boxpoints="outliers",
            )
        )
    fig.update_layout(
        title="Daily Return Distribution by Asset",
        template="plotly_white",
        height=500,
        yaxis_title="Daily Return",
        yaxis_tickformat=".1%",
        showlegend=False,
    )
    return fig


def plot_asset_volatility(
    ticker: str,
    vol_stats: dict[str, pd.DataFrame],
    annual_asset_vol: pd.Series,
) -> go.Figure:
    """Plot rolling, EWMA and GARCH annualised volatility for one asset.

    A horizontal dashed line shows the full-sample annualised historical
    standard deviation supplied in ``annual_asset_vol``.
    """
    if ticker not in vol_stats or ticker not in annual_asset_vol.index:
        raise KeyError(f"No volatility results are available for {ticker}.")
    asset_vol = vol_stats[ticker]
    colors = {"rolling_vol": "#636EFA", "ewma_vol": "#EF553B", "garch_vol": "#00CC96"}
    labels = {"rolling_vol": "Rolling Volatility", "ewma_vol": "EWMA Volatility", "garch_vol": "GARCH Volatility"}

    fig = go.Figure()
    for column in ["rolling_vol", "ewma_vol", "garch_vol"]:
        if column in asset_vol.columns:
            fig.add_trace(
                go.Scatter(
                    x=asset_vol.index,
                    y=asset_vol[column],
                    mode="lines",
                    name=labels[column],
                    line={"color": colors[column], "width": 1.5},
                )
            )
    fig.add_hline(
        y=float(annual_asset_vol[ticker]),
        line_dash="dash",
        line_color="black",
        line_width=1.5,
        annotation_text="Full-sample historical volatility",
        annotation_position="top left",
    )
    fig.update_layout(
        title=f"{ticker} - Annualised Volatility",
        template="plotly_white",
        height=400,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Annualised Volatility",
        yaxis_tickformat=".0%",
        legend={"orientation": "h", "y": 1.15, "x": 0.5, "xanchor": "center"},
        margin={"l": 50, "r": 20, "t": 90, "b": 40},
    )
    return fig


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame) -> go.Figure:
    """Plot an annotated correlation matrix with a fixed [-1, 1] colour scale."""
    fig = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Asset Correlation Matrix",
    )
    fig.update_layout(
        template="plotly_white",
        height=600,
        xaxis_title="Asset",
        yaxis_title="Asset",
        coloraxis_colorbar={"title": "Correlation"},
    )
    return fig


def plot_rolling_correlation(
    rolling_correlation: pd.Series,
    asset1: str,
    asset2: str,
    window: int,
) -> go.Figure:
    """Plot a rolling correlation series for a selected pair of assets."""
    fig = go.Figure(
        go.Scatter(
            x=rolling_correlation.index,
            y=rolling_correlation,
            mode="lines",
            name="Rolling Correlation",
            line={"width": 1.5, "color": "#636EFA"},
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(
        title=f"{window}-Day Rolling Correlation: {asset1} vs {asset2}",
        template="plotly_white",
        height=450,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Correlation",
        yaxis={"range": [-1, 1]},
    )
    return fig


def plot_normality_results(normality_results: pd.DataFrame) -> go.Figure:
    """Plot Jarque-Bera p-values with the selected significance threshold."""
    plot_data = normality_results.reset_index()
    fig = px.bar(
        plot_data,
        x="ticker",
        y="p_value",
        color="normality_not_rejected",
        title="Jarque-Bera Normality Test",
        labels={
            "ticker": "Asset",
            "p_value": "P-value",
            "normality_not_rejected": "Normality not rejected",
        },
    )
    fig.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="5% significance level")
    fig.update_layout(template="plotly_white", height=450, yaxis_title="P-value")
    return fig
