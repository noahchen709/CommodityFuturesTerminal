from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ForecastResult:
    point: float
    lower: float
    upper: float
    interval_width: float
    alpha: float
    probability_up: float | None = None
    signal: int = 0


def make_forecaster_pipeline(alpha: float = 10.0) -> Pipeline:
    """Create the regularized return forecaster used for the conformal range."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def make_direction_pipeline(c: float = 1.0) -> Pipeline:
    """Create a directional classifier for next-week crude returns."""
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    max_iter=1000,
                ),
            ),
        ]
    )


def signal_from_probability(probability_up: float, threshold: float = 0.60) -> int:
    """Convert an up-probability into a long/short/flat signal."""
    lower = 1 - threshold
    if probability_up > threshold:
        return 1
    if probability_up < lower:
        return -1
    return 0


def train_conformal_forecaster(
    df: pd.DataFrame,
    features: list[str],
    target: str = "target_return_1w",
    calibration_size: int = 52,
    alpha: float = 0.2,
    probability_threshold: float = 0.60,
) -> tuple[Pipeline, float, ForecastResult]:
    """Train return and direction models, then create a split-conformal interval."""
    model_df = df.dropna(subset=features + [target]).reset_index(drop=True)
    if len(model_df) <= calibration_size + 30:
        raise ValueError("Need more rows for train/calibration split.")

    train = model_df.iloc[:-calibration_size]
    calibration = model_df.iloc[-calibration_size:]

    model = make_forecaster_pipeline()
    model.fit(train[features], train[target])

    direction_model = make_direction_pipeline()
    direction_model.fit(train[features], (train[target] > 0).astype(int))

    calibration_pred = model.predict(calibration[features])
    residuals = np.abs(calibration[target].to_numpy() - calibration_pred)
    q = float(np.quantile(residuals, 1 - alpha))

    latest_x = df.dropna(subset=features)[features].iloc[[-1]]
    point = float(model.predict(latest_x)[0])
    probability_up = float(direction_model.predict_proba(latest_x)[0, 1])
    result = ForecastResult(
        point=point,
        lower=point - q,
        upper=point + q,
        interval_width=2 * q,
        alpha=alpha,
        probability_up=probability_up,
        signal=signal_from_probability(probability_up, probability_threshold),
    )
    return model, q, result
