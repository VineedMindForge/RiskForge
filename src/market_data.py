"""Market data download and quality-control utilities.

This module is responsible for obtaining historical market prices and
summarising basic data-quality issues. It deliberately does not interpolate
or otherwise modify missing prices: artificial prices can distort returns,
volatility, correlations and Value-at-Risk estimates.

Typical workflow
----------------
>>> prices, quality_report, has_warnings = download_data(
...     ["SPY", "AGG", "QQQ"], period="10y"
... )

The returned ``prices`` DataFrame is intended to be passed to either the
portfolio-construction or market-data-analytics module.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd


TickerInput = Union[str, Iterable[str]]


def validate_market_data(prices: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Create a per-asset data-quality report for a price DataFrame.

    Parameters
    ----------
    prices:
        Adjusted-price DataFrame. The index should contain dates and each
        column should represent one asset/ticker.

    Returns
    -------
    tuple[pandas.DataFrame, bool]
        A report indexed by ticker, containing observation counts, missing
        and invalid price counts, date coverage and duplicate-date counts;
        and a Boolean indicating whether any issue was detected.

    Notes
    -----
    * Missing values and invalid prices are counted separately.
    * Invalid prices are finite values less than or equal to zero, plus
      positive/negative infinity. NaN values are not double-counted here.
    * Duplicate dates are a DataFrame-level issue, so the same count appears
      for every ticker in the report.
    * This function reports issues; it does not clean or repair the data.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    if prices.empty or prices.shape[1] == 0:
        raise ValueError("prices must contain at least one observation and one asset.")

    duplicate_dates = int(prices.index.duplicated().sum())
    report_rows = []
    has_warnings = False

    for ticker in prices.columns:
        asset_prices = pd.to_numeric(prices[ticker], errors="coerce")
        missing_count = int(asset_prices.isna().sum())
        observation_count = int(asset_prices.notna().sum())
        missing_pct = (missing_count / len(asset_prices)) * 100

        finite_mask = np.isfinite(asset_prices.fillna(np.nan))
        invalid_mask = asset_prices.notna() & (
            (asset_prices <= 0) | ~finite_mask
        )
        invalid_count = int(invalid_mask.sum())

        valid_prices = asset_prices[asset_prices.notna() & ~invalid_mask]
        asset_warning = bool(
            missing_count > 0 or invalid_count > 0 or duplicate_dates > 0
        )
        has_warnings = has_warnings or asset_warning

        report_rows.append(
            {
                "ticker": ticker,
                "observations": observation_count,
                "missing_observations": missing_count,
                "missing_percentage": missing_pct,
                "data_completeness_percentage": 100 - missing_pct,
                "invalid_prices": invalid_count,
                "duplicate_dates": duplicate_dates,
                "first_valid_date": valid_prices.index.min() if not valid_prices.empty else pd.NaT,
                "last_valid_date": valid_prices.index.max() if not valid_prices.empty else pd.NaT,
                "latest_valid_price": valid_prices.iloc[-1] if not valid_prices.empty else np.nan,
                "latest_valid_price_date": valid_prices.index[-1] if not valid_prices.empty else pd.NaT,
                "data_validation_warning": asset_warning,
            }
        )

    report = pd.DataFrame(report_rows).set_index("ticker")
    return report, has_warnings


def download_data(
    tickers: TickerInput,
    period: str = "10y",
    interval: str = "1d",
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    auto_adjust: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Download historical market prices using ``yfinance`` and validate them.

    Parameters
    ----------
    tickers:
        A ticker string (for one asset) or an iterable of ticker strings.
    period:
        yfinance period such as ``"1y"``, ``"5y"`` or ``"10y"``. Used when
        neither ``start`` nor ``end`` is supplied.
    interval:
        yfinance interval. The project currently uses daily data (``"1d"``).
    start, end:
        Optional explicit date boundaries accepted by yfinance. If either is
        provided, ``period`` is ignored. yfinance treats ``end`` as exclusive.
    auto_adjust:
        When True, yfinance adjusts prices for splits and distributions.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, bool]
        ``prices``: closing-price DataFrame, sorted by date;
        ``quality_report``: output of :func:`validate_market_data`;
        ``has_warnings``: True when missing/invalid values or duplicate dates
        are present.

    Important
    ---------
    Missing observations are not interpolated or forward-filled. Review the
    quality report and decide how to handle them before using the data in
    portfolio or risk calculations. For multi-asset calculations, all assets
    need a consistent set of usable dates.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Market data downloading requires yfinance. Install it with: pip install yfinance") from exc

    if isinstance(tickers, str):
        ticker_list = [tickers]
    else:
        ticker_list = list(tickers)

    if not ticker_list:
        raise ValueError("Provide at least one ticker.")
    if interval != "1d":
        # Other frequencies may work, but the current analytics annualise
        # using daily trading observations (252 per year).
        raise ValueError("This project currently expects daily data (interval='1d').")

    download_kwargs = {
        "tickers": ticker_list,
        "interval": interval,
        "auto_adjust": auto_adjust,
        "progress": False,
        "group_by": "column",
        "threads": True,
    }
    if start is not None or end is not None:
        download_kwargs["start"] = start
        download_kwargs["end"] = end
    else:
        download_kwargs["period"] = period

    raw = yf.download(**download_kwargs)
    if raw.empty:
        raise ValueError("yfinance returned no market data. Check the tickers and date range.")

    # With group_by='column', Close is normally the top-level field. The
    # fallback handles yfinance versions that return a different column layout.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        elif "Close" in raw.columns.get_level_values(-1):
            prices = raw.xs("Close", axis=1, level=-1).copy()
        else:
            raise KeyError("The downloaded data does not contain a Close field.")
    else:
        if "Close" not in raw.columns:
            raise KeyError("The downloaded data does not contain a Close field.")
        prices = raw[["Close"]].copy()
        prices.columns = ticker_list[:1]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=ticker_list[0])
    elif len(ticker_list) == 1 and prices.shape[1] == 1:
        prices.columns = ticker_list

    prices = prices.sort_index()
    prices.index.name = "Date"
    prices = prices.apply(pd.to_numeric, errors="coerce")

    quality_report, has_warnings = validate_market_data(prices)
    return prices, quality_report, has_warnings


# Backwards-compatible name used in the original notebook.
downloaded_data_validation = validate_market_data
