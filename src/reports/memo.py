from __future__ import annotations

import pandas as pd

from src.models.conformal import ForecastResult


def generate_crude_memo(
    latest: pd.Series,
    forecast: ForecastResult,
    backtest_metrics: dict[str, float],
) -> str:
    """Generate a compact weekly crude oil trading memo."""
    probability_up = forecast.probability_up
    if probability_up is None:
        interval_bias = "constructive" if forecast.point > 0 else "defensive"
    elif forecast.signal > 0:
        interval_bias = "constructive"
    elif forecast.signal < 0:
        interval_bias = "defensive"
    else:
        interval_bias = "neutral"
    crowding = latest["positioning_pct_156w"]
    inventory = latest["inventory_z_52w"]

    if crowding > 0.8:
        positioning_view = "managed money length is crowded"
    elif crowding < 0.2:
        positioning_view = "managed money positioning is washed out"
    else:
        positioning_view = "positioning is balanced"

    if inventory > 1:
        inventory_view = "inventory builds are bearish versus the one-year range"
    elif inventory < -1:
        inventory_view = "inventory draws are supportive versus the one-year range"
    else:
        inventory_view = "inventory changes are near normal"

    probability_view = (
        f"{probability_up:.1%}" if probability_up is not None else "not available"
    )

    return (
        f"WTI setup for {latest['date'].date()}\n\n"
        f"Model view: {interval_bias}. "
        f"Directional probability of an up week is {probability_view}; "
        f"expected next-week return is {forecast.point:.2%}, "
        f"with an {int((1 - forecast.alpha) * 100)}% conformal range of "
        f"{forecast.lower:.2%} to {forecast.upper:.2%}.\n\n"
        f"Supply-demand: {inventory_view}.\n"
        f"Positioning: {positioning_view}.\n"
        f"Risk: annualized 8-week volatility is {latest['vol_8w']:.1%}; recent drawdown is "
        f"{latest['drawdown_13w']:.1%}.\n\n"
        f"Backtest snapshot: Sharpe {backtest_metrics['sharpe']:.2f}, "
        f"max drawdown {backtest_metrics['max_drawdown']:.1%}, "
        f"hit rate {backtest_metrics['hit_rate']:.1%}, "
        f"turnover {backtest_metrics['turnover']:.2f}.\n\n"
        "Suggested setup: size only when the expected return is large relative to the "
        "uncertainty band; avoid leaning into crowded positioning without a fresh "
        "inventory or macro catalyst."
    )
