import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_drawdown(portfolio_value):
    """Calculate portfolio drawdown and summary statistics."""

    running_peak = portfolio_value.cummax()
    drawdown = (portfolio_value / running_peak) - 1

    trough_date = drawdown.idxmin()
    max_drawdown = drawdown.min()

    peak_date = portfolio_value.loc[:trough_date].idxmax()

    recovery_dates = portfolio_value.loc[trough_date:].loc[
        portfolio_value.loc[trough_date:] >= portfolio_value.loc[peak_date]
    ]

    recovery_date = recovery_dates.index[0] if not recovery_dates.empty else None

    current_drawdown = drawdown.iloc[-1]

    return {
        "drawdown": drawdown,
        "running_peak": running_peak,
        "max_drawdown": max_drawdown,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "current_drawdown": current_drawdown
    }

def calculate_drawdown_metrics(portfolio_value, drawdown):
    """Calculate drawdown summary metrics using trading days."""

    max_drawdown = drawdown.min()
    current_drawdown = drawdown.iloc[-1]

    trough_date = drawdown.idxmin()
    peak_date = portfolio_value.loc[:trough_date].idxmax()

    recovery_dates = portfolio_value.loc[trough_date:].loc[
        portfolio_value.loc[trough_date:] >= portfolio_value.loc[peak_date]
    ]

    recovery_date = recovery_dates.index[0] if not recovery_dates.empty else None

    peak_position = portfolio_value.index.get_loc(peak_date)
    trough_position = portfolio_value.index.get_loc(trough_date)

    peak_to_trough_duration = trough_position - peak_position

    peak_to_recovery_duration = (
        portfolio_value.index.get_loc(recovery_date) - peak_position
        if recovery_date is not None else None
    )

    return {
        "max_drawdown": max_drawdown,
        "current_drawdown": current_drawdown,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "peak_to_trough_duration": peak_to_trough_duration,
        "peak_to_recovery_duration": peak_to_recovery_duration,
        "recovery_status": "Recovered" if recovery_date is not None else "Not Recovered"
    }

def plot_drawdown(drawdown):
    """Plot portfolio drawdown over time."""

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown * 100,
        mode="lines",
        name="Drawdown",
        fill="tozeroy"
    ))

    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        height=400,
        margin=dict(l=20, r=20, t=45, b=35)
    )

    return fig


def plot_cumulative_return_drawdown(portfolio_value, drawdown, metrics):
    """Plot cumulative returns and drawdown with peak, trough and recovery markers."""

    cumulative_return = (portfolio_value / portfolio_value.iloc[0] - 1) * 100

    peak_date = metrics["peak_date"]
    trough_date = metrics["trough_date"]
    recovery_date = metrics["recovery_date"]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.6, 0.4],
        subplot_titles=("Cumulative Portfolio Return", "Portfolio Drawdown")
    )

    # Cumulative return
    fig.add_trace(
        go.Scatter(
            x=cumulative_return.index,
            y=cumulative_return,
            mode="lines",
            name="Cumulative Return",
            line=dict(width=2)
        ),
        row=1,
        col=1
    )

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line=dict(width=1.5)
        ),
        row=2,
        col=1
    )

    # Highlight maximum drawdown period
    fig.add_vrect(
        x0=peak_date,
        x1=recovery_date if recovery_date is not None else portfolio_value.index[-1],
        fillcolor="red",
        opacity=0.08,
        line_width=0,
        row="all",
        col=1
    )

    # Peak, trough and recovery markers
    markers = [
        (peak_date, "Peak", "green"),
        (trough_date, "Trough", "red")
    ]

    if recovery_date is not None:
        markers.append((recovery_date, "Recovery", "blue"))

    for date, label, color in markers:
        fig.add_vline(
            x=date,
            line_dash="dash",
            line_color=color,
            line_width=1.5,
            row="all",
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=label,
                line=dict(color=color, dash="dash", width=1.5)
            )
        )

    # Layout
    fig.update_layout(
        height=700,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=90, b=35),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_yaxes(title_text="Cumulative Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig