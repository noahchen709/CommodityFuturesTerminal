from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.conformal import (
    make_direction_pipeline,
    make_forecaster_pipeline,
    signal_from_probability,
)


def run_walk_forward_backtest(
    df: pd.DataFrame,
    features: list[str],
    target: str = "target_return_1w",
    train_window: int = 156,
    probability_threshold: float = 0.60,
) -> pd.DataFrame:
    """Walk-forward backtest using directional probability for trade signals."""
    data = df.dropna(subset=features + [target]).reset_index(drop=True)
    rows: list[dict[str, float | pd.Timestamp]] = []

    for idx in range(train_window, len(data)):
        train = data.iloc[idx - train_window : idx]
        current = data.iloc[[idx]]

        return_model = make_forecaster_pipeline()
        return_model.fit(train[features], train[target])
        forecast = float(return_model.predict(current[features])[0])

        direction_model = make_direction_pipeline()
        direction_model.fit(train[features], (train[target] > 0).astype(int))
        probability_up = float(direction_model.predict_proba(current[features])[0, 1])
        signal = signal_from_probability(probability_up, probability_threshold)

        realized = float(data[target].iloc[idx])
        rows.append(
            {
                "date": data["date"].iloc[idx],
                "forecast": forecast,
                "probability_up": probability_up,
                "entry_threshold": probability_threshold,
                "signal": signal,
                "realized_return": realized,
                "strategy_return": signal * realized,
            }
        )

    return pd.DataFrame(rows)


def summarize_backtest(results: pd.DataFrame) -> dict[str, float]:
    if results.empty:
        return {"sharpe": 0.0, "max_drawdown": 0.0, "hit_rate": 0.0, "turnover": 0.0}

    strategy = results["strategy_return"].fillna(0)
    equity = (1 + strategy).cumprod()
    drawdown = equity / equity.cummax() - 1
    active = results["signal"] != 0
    hits = np.sign(results.loc[active, "strategy_return"]) > 0

    weekly_vol = strategy.std()
    sharpe = 0.0 if weekly_vol == 0 else float(strategy.mean() / weekly_vol * np.sqrt(52))
    return {
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float(hits.mean()) if len(hits) else 0.0,
        "turnover": float(results["signal"].diff().abs().fillna(0).mean()),
    }
