"""Historical and custom-scenario stress testing for a portfolio."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def identify_worst_portfolio_scenarios(portfolio_returns, portfolio_value, horizons=(1, 5, 20), top_n=5):
    """Identify the worst historical portfolio return windows."""

    if not np.isfinite(portfolio_value) or portfolio_value < 0:
        raise ValueError("portfolio_value must be finite and non-negative.")

    if top_n < 1 or int(top_n) != top_n:
        raise ValueError("top_n must be a positive integer.")

    if not horizons or any(h < 1 or int(h) != h for h in horizons):
        raise ValueError("horizons must contain positive integers.")

    if portfolio_returns.isna().any() or not np.isfinite(portfolio_returns).all():
        raise ValueError("Portfolio returns must contain finite values.")

    if (portfolio_returns <= -1).any():
        raise ValueError("Portfolio returns cannot be below or equal to -100%.")

    if not portfolio_returns.index.is_unique or not portfolio_returns.index.is_monotonic_increasing:
        raise ValueError("Portfolio returns must have unique dates sorted in ascending order.")

    results = {}

    for horizon in horizons:
        if len(portfolio_returns) < horizon:
            raise ValueError(f"Insufficient observations for the {horizon}-day horizon.")

        rolling_returns = (
            (1 + portfolio_returns)
            .rolling(horizon)
            .apply(np.prod, raw=True) - 1
        ).dropna()

        worst_scenarios = rolling_returns.nsmallest(top_n)
        scenario_results = []

        for rank, (end_date, stress_return) in enumerate(worst_scenarios.items(), start=1):
            end_position = portfolio_returns.index.get_loc(end_date)
            start_date = portfolio_returns.index[end_position - horizon + 1]

            stress_pnl = portfolio_value * stress_return

            scenario_results.append({
                "Rank": rank,
                "Start Date": start_date,
                "End Date": end_date,
                "Stress Return": stress_return,
                "Stress P&L": stress_pnl,
                "Stressed Portfolio Value": portfolio_value + stress_pnl
            })

        results[horizon] = pd.DataFrame(scenario_results).set_index("Rank")

    return results


def calculate_stress_impact(weights, shocks, portfolio_value):
    """Calculate portfolio return, P&L and value after an asset shock scenario."""

    weights = pd.Series(weights, dtype=float)
    shocks = pd.Series(shocks, dtype=float)

    if not np.isfinite(portfolio_value) or portfolio_value < 0:
        raise ValueError("portfolio_value must be finite and non-negative.")

    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Weights must be finite and non-negative.")

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")

    if not np.isfinite(shocks).all() or (shocks < -1).any():
        raise ValueError("Shocks must be finite and cannot be below -100%.")

    unknown_assets = shocks.index.difference(weights.index)
    if len(unknown_assets) > 0:
        raise ValueError(f"Unknown assets in shocks: {unknown_assets.tolist()}")

    shocks = shocks.reindex(weights.index, fill_value=0)
    portfolio_stress_return = (weights * shocks).sum()
    stress_pnl = portfolio_value * portfolio_stress_return

    return {
        "stress_return": portfolio_stress_return,
        "stress_pnl": stress_pnl,
        "stressed_portfolio_value": portfolio_value + stress_pnl
    }


def run_stress_scenarios(weights, scenarios, portfolio_value):
    """Apply a dictionary of named asset-shock scenarios to the portfolio."""

    results = []

    for scenario_name, shocks in scenarios.items():
        stress_result = calculate_stress_impact(weights, shocks, portfolio_value)

        results.append({
            "Scenario": scenario_name,
            "Stress Return": stress_result["stress_return"],
            "Stress P&L": stress_result["stress_pnl"],
            "Stressed Portfolio Value": stress_result["stressed_portfolio_value"]
        })

    if not results:
        return pd.DataFrame(columns=["Stress Return", "Stress P&L", "Stressed Portfolio Value"])

    return pd.DataFrame(results).set_index("Scenario")


def create_custom_scenario(weights, shocks):
    """Create a complete shock dictionary, assigning zero to unspecified assets."""

    weights = pd.Series(weights, dtype=float)
    shocks = pd.Series(shocks, dtype=float)

    unknown_assets = shocks.index.difference(weights.index)
    if len(unknown_assets) > 0:
        raise ValueError(f"Unknown assets in shocks: {unknown_assets.tolist()}")

    if not np.isfinite(shocks).all() or (shocks < -1).any():
        raise ValueError("Shocks must be finite and cannot be below -100%.")

    return shocks.reindex(weights.index, fill_value=0).to_dict()


def identify_historical_scenarios(asset_returns, weights, horizons=(1, 5, 20), top_n=5):
    """Find the worst historical asset-return windows for the current portfolio.

    Each historical asset shock is compounded over the selected horizon. The
    shocks are then applied to today's weights. Complete common dates are used.
    """

    weights = pd.Series(weights, dtype=float)

    if top_n < 1 or int(top_n) != top_n:
        raise ValueError("top_n must be a positive integer.")

    if not horizons or any(h < 1 or int(h) != h for h in horizons):
        raise ValueError("horizons must contain positive integers.")

    if not weights.index.is_unique or not asset_returns.columns.is_unique:
        raise ValueError("Asset names must be unique.")

    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Weights must be finite, non-negative and sum to 1.")

    if not asset_returns.index.is_unique or not asset_returns.index.is_monotonic_increasing:
        raise ValueError("asset_returns must have unique dates sorted in ascending order.")

    missing_assets = weights.index.difference(asset_returns.columns)
    if len(missing_assets) > 0:
        raise ValueError(f"Missing return data for assets: {missing_assets.tolist()}")

    returns = asset_returns.reindex(columns=weights.index).dropna(how="any")

    if not np.isfinite(returns.to_numpy()).all() or (returns <= -1).any().any():
        raise ValueError("Asset returns must be finite and greater than -100%.")

    results = {}

    for horizon in horizons:
        if len(returns) < horizon:
            raise ValueError(f"Insufficient observations for the {horizon}-day horizon.")

        asset_shocks = (1 + returns).rolling(horizon).apply(np.prod, raw=True) - 1
        portfolio_shocks = asset_shocks.mul(weights, axis=1).sum(axis=1)
        worst_scenarios = portfolio_shocks.nsmallest(top_n)
        scenario_results = []

        for rank, (end_date, portfolio_return) in enumerate(worst_scenarios.items(), start=1):
            end_position = returns.index.get_loc(end_date)
            start_date = returns.index[end_position - horizon + 1]

            row = {
                "Rank": rank,
                "Start Date": start_date,
                "End Date": end_date,
                "Portfolio Stress Return": portfolio_return
            }

            for asset in weights.index:
                row[f"{asset} Shock"] = asset_shocks.loc[end_date, asset]

            scenario_results.append(row)

        results[horizon] = pd.DataFrame(scenario_results).set_index("Rank")

    return results


def calculate_asset_stress_contributions(weights, shocks, portfolio_value):
    """Show each asset's contribution to the total stress P&L."""

    weights = pd.Series(weights, dtype=float)
    shocks = pd.Series(shocks, dtype=float)

    unknown_assets = shocks.index.difference(weights.index)
    if len(unknown_assets) > 0:
        raise ValueError(f"Unknown assets in shocks: {unknown_assets.tolist()}")

    if not np.isfinite(weights).all() or not np.isfinite(shocks).all():
        raise ValueError("Weights and shocks must contain finite values.")

    if (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Weights must be non-negative and sum to 1.")

    if (shocks < -1).any():
        raise ValueError("Asset price shocks cannot be below -100%.")

    if not np.isfinite(portfolio_value) or portfolio_value < 0:
        raise ValueError("portfolio_value must be finite and non-negative.")

    shocks = shocks.reindex(weights.index, fill_value=0)
    asset_pnl = portfolio_value * weights * shocks
    total_pnl = asset_pnl.sum()

    if total_pnl != 0:
        contribution_percentage = asset_pnl / total_pnl * 100
    else:
        contribution_percentage = pd.Series(np.nan, index=weights.index)

    contributions = pd.DataFrame({
        "Weight": weights,
        "Stress Shock": shocks,
        "Stress P&L": asset_pnl,
        "Contribution (%)": contribution_percentage
    })

    return contributions.sort_values("Stress P&L")

def calculate_historical_stress_attribution(asset_returns, portfolio_returns, allocation_drift, start_date, end_date, portfolio_value):
    """Calculate exact asset-level P&L contributions during a historical stress period."""

    asset_returns = asset_returns.loc[start_date:end_date]
    portfolio_returns = portfolio_returns.loc[start_date:end_date]
    allocation_drift = allocation_drift.loc[start_date:end_date]

    common_dates = portfolio_returns.index.intersection(asset_returns.index).intersection(allocation_drift.index)

    asset_returns = asset_returns.loc[common_dates]
    portfolio_returns = portfolio_returns.loc[common_dates]
    allocation_drift = allocation_drift.loc[common_dates]

    if asset_returns.empty:
        raise ValueError("No matching return observations found for the selected period.")

    if not np.isfinite(asset_returns.to_numpy()).all() or not np.isfinite(portfolio_returns.to_numpy()).all():
        raise ValueError("Returns must contain finite values.")

    if not np.isfinite(allocation_drift.to_numpy()).all():
        raise ValueError("Allocation weights must be finite.")

    if not np.allclose(
        (asset_returns * allocation_drift).sum(axis=1),
        portfolio_returns,
        atol=1e-8
    ):
        raise ValueError("Asset returns and allocation weights do not reconcile with portfolio returns.")

    # Portfolio value at the beginning of each daily return period
    daily_starting_value = portfolio_value * (1 + portfolio_returns).cumprod().shift(1).fillna(1)

    # Calculate each asset's daily monetary contribution
    daily_asset_pnl = asset_returns * allocation_drift.mul(daily_starting_value, axis=0)

    # Aggregate contributions across the stress period
    asset_pnl = daily_asset_pnl.sum()

    total_stress_pnl = portfolio_value * ((1 + portfolio_returns).prod() - 1)

    contributions = pd.DataFrame({
        "Asset Return": (1 + asset_returns).prod() - 1,
        "Stress Contribution": asset_pnl / portfolio_value,
        "Stress P&L": asset_pnl,
        "Contribution (%)": asset_pnl / total_stress_pnl * 100 if total_stress_pnl != 0 else np.nan
    })

    return contributions.sort_values("Stress P&L")

def plot_asset_stress_attribution(attribution):
    """Compare individual asset returns and portfolio return contributions."""

    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=attribution.index,
        y=attribution["Asset Return"] * 100,
        name="Asset Return",
        text=attribution["Asset Return"].map(lambda x: f"{x:.2%}"),
        textposition="outside"
    ))

    fig.add_trace(go.Bar(
        x=attribution.index,
        y=attribution["Stress Contribution"] * 100,
        name="Portfolio Contribution",
        text=attribution["Stress Contribution"].map(lambda x: f"{x:.2%}"),
        textposition="outside"
    ))

    fig.update_layout(
        title="Asset Returns vs Portfolio Return Contributions",
        template="plotly_white",
        xaxis_title="Asset",
        yaxis_title="Return / Contribution (%)",
        barmode="group",
        height=330,
        legend_title=None
    )

    fig.update_yaxes(ticksuffix="%")

    return fig

def plot_stress_pnl_attribution(attribution):
    """Plot monetary stress P&L contributions by asset."""

    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=attribution["Stress P&L"],
        y=attribution.index,
        orientation="h",
        text=attribution["Stress P&L"].map(lambda x: f"${x:,.2f}"),
        textposition="outside",
        marker_color="#EF553B"
    ))

    fig.update_layout(
        title="Asset Contribution to Portfolio Stress Loss",
        template="plotly_white",
        xaxis_title="Stress P&L ($)",
        yaxis_title=None,
        height=280,
        showlegend=False
    )

    fig.update_xaxes(tickprefix="$")

    return fig