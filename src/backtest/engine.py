from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.conformal import make_forecaster_pipeline


def run_walk_forward_backtest(
    df: pd.DataFrame,
    features: list[str],
    target: str = "target_return_1w",
    train_window: int = 156,
    risk_threshold: float = 0.25,
) -> pd.DataFrame:
    """Walk-forward backtest using the same forecaster as the live forecast."""
    data = df.dropna(subset=features + [target]).reset_index(drop=True)
    rows: list[dict[str, float | pd.Timestamp]] = []

    for idx in range(train_window, len(data) - 1):
        train = data.iloc[idx - train_window : idx]
        current = data.iloc[[idx]]

        model = make_forecaster_pipeline()
        model.fit(train[features], train[target])
        forecast = float(model.predict(current[features])[0])
        vol = max(float(current["vol_8w"].iloc[0] / np.sqrt(52)), 0.005)

        if forecast > risk_threshold * vol:
            signal = 1
        elif forecast < -risk_threshold * vol:
            signal = -1
        else:
            signal = 0

        realized = float(data[target].iloc[idx])
        rows.append(
            {
                "date": data["date"].iloc[idx],
                "forecast": forecast,
                "entry_threshold": risk_threshold * vol,
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
