from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ForecastResult:
    point: float
    lower: float
    upper: float
    interval_width: float
    alpha: float


def train_conformal_forecaster(
    df: pd.DataFrame,
    features: list[str],
    target: str = "target_return_1w",
    calibration_size: int = 52,
    alpha: float = 0.2,
) -> tuple[Pipeline, float, ForecastResult]:
    """Train a model and create a split-conformal interval for the latest row."""
    model_df = df.dropna(subset=features + [target]).reset_index(drop=True)
    if len(model_df) <= calibration_size + 30:
        raise ValueError("Need more rows for train/calibration split.")

    train = model_df.iloc[:-calibration_size]
    calibration = model_df.iloc[-calibration_size:]

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=8,
                    random_state=13,
                ),
            ),
        ]
    )
    model.fit(train[features], train[target])

    calibration_pred = model.predict(calibration[features])
    residuals = np.abs(calibration[target].to_numpy() - calibration_pred)
    q = float(np.quantile(residuals, 1 - alpha))

    latest_x = df.dropna(subset=features)[features].iloc[[-1]]
    point = float(model.predict(latest_x)[0])
    result = ForecastResult(
        point=point,
        lower=point - q,
        upper=point + q,
        interval_width=2 * q,
        alpha=alpha,
    )
    return model, q, result
