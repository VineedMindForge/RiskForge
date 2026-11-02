"""Historical and custom-scenario stress testing for a portfolio."""

import numpy as np
import pandas as pd


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
