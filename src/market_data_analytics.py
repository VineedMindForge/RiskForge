"""Asset returns, statistics, volatility, correlation and distribution analysis."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import jarque_bera, norm


TRADING_DAYS = 252


def calculate_asset_statistics(prices):
    """Summarise each asset's price history and missing observations."""

    asset_data = []

    for ticker in prices.columns:
        asset_prices = prices[ticker]
        missing = asset_prices.isna().sum()
        observations = asset_prices.notna().sum()
        missing_pct = missing / len(prices) * 100 if len(prices) else 0
        invalid = asset_prices.notna() & ((asset_prices <= 0) | ~np.isfinite(asset_prices))
        valid_prices = asset_prices[asset_prices.notna() & ~invalid]

        asset_data.append({
            "ticker": ticker,
            "first_date": valid_prices.index.min() if not valid_prices.empty else pd.NaT,
            "last_date": valid_prices.index.max() if not valid_prices.empty else pd.NaT,
            "missing_observations": missing,
            "missing_percentage": missing_pct,
            "number_of_observations": observations,
            "data_completeness_percentage": 100 - missing_pct,
            "latest_valid_price": valid_prices.iloc[-1] if not valid_prices.empty else np.nan,
            "latest_valid_price_date": valid_prices.index[-1] if not valid_prices.empty else pd.NaT,
            "invalid_prices": invalid.sum(),
            "data_validation_warning": missing > 0 or invalid.any()
        })

    return pd.DataFrame(asset_data).set_index("ticker")


def calculate_asset_returns(prices):
    """Calculate simple daily returns for each asset."""

    return prices.pct_change(fill_method=None)


def calculate_return_statistics(daily_returns):
    """Calculate average return, volatility, skewness and kurtosis by asset."""

    asset_data = []

    for ticker in daily_returns.columns:
        returns = daily_returns[ticker].replace([np.inf, -np.inf], np.nan).dropna()

        asset_data.append({
            "ticker": ticker,
            "daily_mean_return": returns.mean(),
            "annualized_return": returns.mean() * TRADING_DAYS,
            "daily_volatility": returns.std(),
            "annualized_volatility": returns.std() * np.sqrt(TRADING_DAYS),
            "skewness": returns.skew(),
            "excess_kurtosis": returns.kurt(),
            "best_daily_return": returns.max(),
            "worst_daily_return": returns.min()
        })

    return pd.DataFrame(asset_data).set_index("ticker")


def calculate_rolling_vol(asset_return, window=30):
    """Calculate annualised rolling volatility."""

    return asset_return.rolling(window).std() * np.sqrt(TRADING_DAYS)


def calculate_ewma_vol(asset_return, lambda_=0.94, window=30):
    """Calculate annualised EWMA volatility using squared daily returns."""

    ewma_variance = asset_return.pow(2).ewm(
        alpha=1 - lambda_, min_periods=window, adjust=False
    ).mean()
    return np.sqrt(ewma_variance) * np.sqrt(TRADING_DAYS)


def calculate_garch_vol(asset_return):
    """Fit a GARCH(1,1) model and return annualised conditional volatility."""

    from arch import arch_model

    returns = asset_return.dropna() * 100
    model = arch_model(returns, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
    fitted_model = model.fit(disp="off")
    garch_volatility = fitted_model.conditional_volatility / 100
    return garch_volatility * np.sqrt(TRADING_DAYS)


def vol_analysis(daily_returns, window=30, lambda_=0.94):
    """Calculate rolling, EWMA and GARCH volatility for every selected asset."""

    vol_stats = {}

    for ticker in daily_returns.columns:
        returns = daily_returns[ticker]
        rolling_vol = calculate_rolling_vol(returns, window)
        ewma_vol = calculate_ewma_vol(returns, lambda_, window)
        garch_vol = calculate_garch_vol(returns)

        vol_stats[ticker] = pd.concat([
            rolling_vol.rename("rolling_vol"),
            ewma_vol.rename("ewma_vol"),
            garch_vol.rename("garch_vol")
        ], axis=1)

    annual_asset_vol = daily_returns.std() * np.sqrt(TRADING_DAYS)
    return vol_stats, annual_asset_vol


def calculate_correlation_matrix(daily_returns, window=None):
    """Calculate the correlation matrix using all data or the latest N rows."""

    if window is not None:
        daily_returns = daily_returns.tail(window)
    return daily_returns.corr()


def calculate_rolling_correlation(daily_returns, asset1, asset2, window=60):
    """Calculate rolling correlation for two selected assets."""

    return daily_returns[asset1].rolling(window).corr(daily_returns[asset2])


def calculate_normality_test(asset_returns, significance_level=0.05):
    """Run the Jarque-Bera test for each asset's daily return distribution."""

    normality = []

    for ticker in asset_returns.columns:
        returns = asset_returns[ticker].dropna()
        statistic, p_value = jarque_bera(returns)

        normality.append({
            "ticker": ticker,
            "observations": len(returns),
            "jb_statistic": statistic,
            "p_value": p_value,
            "normality_not_rejected": p_value >= significance_level
        })

    return pd.DataFrame(normality).set_index("ticker")


def plot_asset_returns(ticker, daily_returns):
    """Plot the daily return time series for one asset."""

    returns = daily_returns[ticker].dropna()
    fig = px.line(x=returns.index, y=returns, title=f"{ticker} - Daily Returns")
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(template="plotly_white", hovermode="x unified",
                      xaxis_title="Date", yaxis_title="Daily Return", yaxis_tickformat=".1%")
    return fig


def plot_return_distribution(ticker, daily_returns):
    """Plot a return histogram with a normal-distribution reference curve."""

    returns = daily_returns[ticker].dropna()
    mean = returns.mean()
    standard_deviation = returns.std()
    x_values = np.linspace(returns.min(), returns.max(), 300)
    normal_curve = norm.pdf(x_values, mean, standard_deviation)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns, histnorm="probability density", name="Observed Returns"))
    fig.add_trace(go.Scatter(x=x_values, y=normal_curve, mode="lines", name="Normal Distribution"))
    fig.add_vline(x=mean, line_dash="dash", line_color="black", annotation_text="Mean")
    fig.update_layout(template="plotly_white", title=f"{ticker} - Daily Return Distribution",
                      xaxis_title="Daily Return", yaxis_title="Density", xaxis_tickformat=".1%")
    return fig


def plot_return_boxplot(daily_returns):
    """Compare daily return distributions across the selected assets."""

    fig = go.Figure()
    for ticker in daily_returns.columns:
        fig.add_trace(go.Box(y=daily_returns[ticker].dropna(), name=ticker))
    fig.update_layout(template="plotly_white", title="Daily Return Distribution by Asset",
                      yaxis_title="Daily Return", yaxis_tickformat=".1%", showlegend=False)
    return fig


def plot_asset_volatility(ticker, vol_stats, annual_asset_vol):
    """Plot rolling, EWMA and GARCH volatility for one asset."""

    asset_volatility = vol_stats[ticker]
    fig = px.line(asset_volatility, x=asset_volatility.index,
                  y=["rolling_vol", "ewma_vol", "garch_vol"],
                  title=f"{ticker} - Annualised Volatility")
    fig.add_hline(y=annual_asset_vol[ticker], line_dash="dash", line_color="black",
                  annotation_text="Full-sample historical volatility")
    fig.update_layout(template="plotly_white", hovermode="x unified",
                      xaxis_title="Date", yaxis_title="Annualised Volatility", yaxis_tickformat=".0%")
    return fig


def plot_correlation_heatmap(correlation_matrix):
    """Plot an annotated asset correlation matrix."""

    fig = px.imshow(correlation_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title="Asset Correlation Matrix")
    fig.update_layout(template="plotly_white", height=600)
    return fig


def plot_rolling_correlation(rolling_corr, asset1, asset2, window):
    """Plot rolling correlation for a selected pair of assets."""

    fig = px.line(x=rolling_corr.index, y=rolling_corr,
                  title=f"{window}-Day Rolling Correlation: {asset1} vs {asset2}")
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(template="plotly_white", hovermode="x unified",
                      xaxis_title="Date", yaxis_title="Correlation", yaxis_range=[-1, 1])
    return fig
