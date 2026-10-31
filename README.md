# Market Risk Project — reusable Python modules

These files are the calculation and chart layer extracted from `market_modules.ipynb`.
They are **not** the Streamlit pages. The Streamlit pages will collect user inputs,
call these functions, and display the returned values and Plotly figures.

## File responsibilities

### `market_data.py`
Shared market-data ingestion used by both application pages.

- `download_data()` downloads adjusted daily closing prices with yfinance.
- `validate_market_data()` reports missing observations, invalid prices and duplicate dates.
- `downloaded_data_validation` is retained as a backwards-compatible alias.

Do not interpolate financial prices automatically. The module reports missing
observations but leaves the decision about alignment or exclusion to the caller.

### `portfolio_construction.py`
Calculations and visualizations for Page 1 — Portfolio Construction.

- `calculate_portfolio_returns()` supports Daily, Monthly and Buy and Hold.
- It returns both portfolio returns and beginning-of-day allocation weights.
- `calculate_portfolio_value()` compounds returns and includes the initial
  investment at the supplied initial date.
- `calculate_returns_metrics()`, `calculate_volatility_metrics()` and
  `calculate_portfolio_metrics()` calculate portfolio performance statistics.
- `plot_portfolio_vs_assets()`, `plot_allocation_over_time()` and
  `plot_current_allocation()` return Plotly figures.

### `market_data_analytics.py`
Individual-asset calculations and visualizations for Page 2 — Market Data & Analytics.

- Asset and return statistics.
- Rolling, EWMA and GARCH volatility.
- Correlation matrix and selected-pair rolling correlation.
- Jarque-Bera normality test.
- Plotly charts for prices, returns, distributions, volatility, correlations
  and normality-test p-values.

Covariance matrix, Ljung-Box/autocorrelation test and Q-Q plot data are not
included, per the current project scope. A covariance matrix may still be
needed internally by later parametric portfolio-risk calculations.

## Install dependencies

In the project's activated virtual environment:

```bash
python -m pip install pandas numpy yfinance scipy plotly arch streamlit
```

`arch` is only needed for GARCH calculations. The GARCH import is lazy, so the
other analytics can be used without loading that package until GARCH is called.

## Suggested project layout

```text
quantitative-risk-model-validation/
│
├── app.py
├── pages/
│   ├── 1_Portfolio_Construction.py
│   └── 2_Market_Data_Analytics.py
├── src/
│   ├── __init__.py
│   ├── market_data.py
│   ├── portfolio_construction.py
│   └── market_data_analytics.py
└── notebooks/
    └── market_modules.ipynb
```

Copy these three modules into the project's `src/` directory. Keep notebook-only
controls such as `ipywidgets.interact()` out of the modules. In Streamlit, use
`st.selectbox`, `st.slider`, `st.date_input`, etc. to collect inputs.

## Example: download data and build a portfolio

```python
import pandas as pd

from src.market_data import download_data
from src.portfolio_construction import (
    calculate_portfolio_returns,
    calculate_portfolio_value,
    calculate_portfolio_metrics,
    plot_portfolio_vs_assets,
    plot_allocation_over_time,
    plot_current_allocation,
)

# Download a common 10-year daily price history.
tickers = ["SPY", "AGG", "QQQ"]
prices, data_quality, has_warnings = download_data(tickers, period="10y")

# Select the dates used for the portfolio simulation.
investment_start = "2021-01-01"
investment_end = "2025-12-31"
investment_prices = prices.loc[investment_start:investment_end].copy()

# The calculation module requires complete, positive prices for all selected
# assets on all selected dates. Do not silently interpolate missing prices.
investment_prices = investment_prices.dropna(how="any")

weights = pd.Series({"SPY": 0.50, "AGG": 0.30, "QQQ": 0.20})
portfolio_returns, allocation_drift = calculate_portfolio_returns(
    investment_prices, weights, rebalance_frequency="Monthly"
)

portfolio_value = calculate_portfolio_value(
    portfolio_returns,
    initial_investment=100_000,
    initial_date=investment_prices.index[0],
)

metrics = calculate_portfolio_metrics(portfolio_value, risk_free_rate=0.04)

# Notebook display:
plot_portfolio_vs_assets(investment_prices, portfolio_value).show()
plot_allocation_over_time(allocation_drift).show()
plot_current_allocation(allocation_drift).show()
print(metrics)
```

## Example: market analytics

```python
from src.market_data_analytics import (
    calculate_asset_statistics,
    calculate_asset_returns,
    calculate_return_statistics,
    vol_analysis,
    calculate_correlation_matrix,
    calculate_rolling_correlation,
    calculate_normality_test,
    plot_asset_volatility,
    plot_correlation_heatmap,
    plot_rolling_correlation,
)

# Use the full downloaded history for market data analysis.
daily_returns = calculate_asset_returns(prices)
asset_statistics = calculate_asset_statistics(prices)
return_statistics = calculate_return_statistics(daily_returns)
vol_stats, annual_asset_vol = vol_analysis(daily_returns, window=30, lambda_=0.94)
correlation_matrix = calculate_correlation_matrix(daily_returns)
normality_results = calculate_normality_test(daily_returns)

# Notebook display examples:
plot_asset_volatility("SPY", vol_stats, annual_asset_vol).show()
plot_correlation_heatmap(correlation_matrix).show()
rolling = calculate_rolling_correlation(daily_returns, "SPY", "AGG", window=60)
plot_rolling_correlation(rolling, "SPY", "AGG", window=60).show()
```

## Use in a Streamlit page

A calculation function returns a figure; it does not display it. The page decides
where it appears:

```python
import streamlit as st
from src.portfolio_construction import plot_allocation_over_time

fig = plot_allocation_over_time(allocation_drift)
st.plotly_chart(fig, use_container_width=True)
```

## Methodology assumptions to be aware of

1. All portfolio functions assume long-only weights that sum to 1.
2. All assets in a portfolio simulation must have complete, valid prices on the
   same dates. The example drops rows with any missing asset price; inspect the
   data-quality report first and be aware this shortens the sample.
3. Daily rebalancing means target weights are restored before each day's return.
4. Monthly rebalancing means target weights are restored before the first
   available trading day's return in each new calendar month.
5. Buy and Hold means initial holdings are retained and weights drift with prices.
6. Rebalancing ignores commissions, taxes, slippage and market impact.
7. Historical volatility, rolling volatility, EWMA and GARCH outputs are
   annualised using `sqrt(252)`.
8. `annual_asset_vol` in `vol_analysis()` is the full-sample historical
   annualised volatility and is intended as a constant reference line.
9. The Sharpe ratio uses annualised arithmetic mean daily return less the
   annual risk-free rate, divided by annualised volatility.
10. The Jarque-Bera output `normality_not_rejected=True` means the null of
    normality was not rejected at the selected significance level; it does not
    establish that returns are normally distributed.
