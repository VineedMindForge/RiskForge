"""Download up to 10 years of market prices and check data quality."""

import numpy as np
import pandas as pd


def downloaded_data_validation(prices):
    """Return a data-quality report and a warning flag for each asset."""

    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty DataFrame.")

    prices = prices.sort_index()
    report = []
    warning = False
    duplicate_dates = int(prices.index.duplicated().sum())

    for ticker in prices.columns:
        asset_prices = pd.to_numeric(prices[ticker], errors="coerce")

        valid_mask = (
            asset_prices.notna()
            & np.isfinite(asset_prices)
            & (asset_prices > 0)
        )
        valid_prices = asset_prices[valid_mask]

        if valid_prices.empty:
            first_date = pd.NaT
            last_date = pd.NaT
            observations = 0
            missing = 0
            missing_pct = 0.0
            invalid = int((asset_prices.notna() & ~np.isfinite(asset_prices)).sum() + (asset_prices.notna() & (asset_prices <= 0)).sum())
            latest_price = np.nan
            latest_price_date = pd.NaT
            asset_warning = True
        else:
            first_date = valid_prices.index[0]
            last_date = valid_prices.index[-1]

            # Only count missing observations between the asset's first and last
            # available prices. Dates before listing are expected, not errors.
            available_period = asset_prices.loc[first_date:last_date]
            observations = int(valid_prices.loc[first_date:last_date].count())
            missing = int(available_period.isna().sum())
            missing_pct = missing / len(available_period) * 100
            invalid = int(
                (available_period.notna() & ((available_period <= 0) | ~np.isfinite(available_period))).sum()
            )
            latest_price = float(valid_prices.iloc[-1])
            latest_price_date = valid_prices.index[-1]
            asset_warning = missing > 0 or invalid > 0 or duplicate_dates > 0

        warning = warning or asset_warning

        report.append({
            "Ticker": ticker,
            "Observations": observations,
            "Missing": missing,
            "Missing %": missing_pct,
            "Invalid Prices": invalid,
            "Duplicate Dates": duplicate_dates,
            "First Date": first_date,
            "Last Date": last_date,
            "Latest Valid Price": latest_price,
            "Latest Valid Price Date": latest_price_date,
            "Data Warning": asset_warning
        })

    report = pd.DataFrame(report).set_index("Ticker")
    return report, warning


def download_data(tickers, period="10y", interval="1d"):
    """Download adjusted closing prices for the requested period (10 years by default).

    Each asset is returned with whatever history is available. A recently listed
    asset will have missing values before its first available trading date.
    Those leading values are retained and are not treated as data-quality errors.
    Missing prices are never interpolated or forward-filled.
    """

    import yfinance as yf

    if isinstance(tickers, str):
        ticker_list = [tickers]
    else:
        ticker_list = list(tickers)

    if not ticker_list:
        raise ValueError("Select at least one ticker.")

    if interval != "1d":
        raise ValueError("This project currently uses daily market data (interval='1d').")

    data = yf.download(
        tickers=ticker_list,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError("No market data was downloaded. Check the selected tickers.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        elif "Close" in data.columns.get_level_values(-1):
            prices = data.xs("Close", axis=1, level=-1).copy()
        else:
            raise ValueError("The downloaded data does not contain closing prices.")
    else:
        if "Close" not in data.columns:
            raise ValueError("The downloaded data does not contain closing prices.")
        prices = data[["Close"]].copy()
        prices.columns = [ticker_list[0]]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=ticker_list[0])

    # Keep all requested columns. A ticker with no returned history will have
    # an empty column and will be identified in the quality report.
    prices = prices.reindex(columns=ticker_list)
    prices = prices.sort_index()
    prices.index.name = "Date"
    prices = prices.apply(pd.to_numeric, errors="coerce")

    quality_report, has_warnings = downloaded_data_validation(prices)
    return prices, quality_report, has_warnings
