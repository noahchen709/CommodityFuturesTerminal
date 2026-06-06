from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import ElasticNet, Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.conformal import (
    make_direction_pipeline,
    make_forecaster_pipeline,
    signal_from_probability,
)


RIDGE_ALPHAS = np.array([0.1, 1.0, 3.0, 10.0, 30.0, 100.0])


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


def run_model_leaderboard_backtest(
    df: pd.DataFrame,
    features: list[str],
    target: str = "target_return_1w",
    train_window: int = 156,
    probability_threshold: float = 0.60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare baseline, linear, classifier, tree, and rule strategies walk-forward."""
    data = df.dropna(subset=features + [target]).reset_index(drop=True)
    rows: list[dict[str, float | int | str | pd.Timestamp | None]] = []

    for idx in range(train_window, len(data)):
        train = data.iloc[idx - train_window : idx]
        current = data.iloc[[idx]]
        realized = float(data[target].iloc[idx])

        step_rows = []
        step_rows.extend(_baseline_predictions(train, current, target, realized))
        step_rows.extend(_regression_predictions(train, current, features, target, realized))
        step_rows.extend(
            _classifier_predictions(
                train,
                current,
                features,
                target,
                realized,
                probability_threshold,
            )
        )
        step_rows.extend(_rule_predictions(train, current, target, realized))

        for row in step_rows:
            row["date"] = data["date"].iloc[idx]
        rows.extend(step_rows)

    predictions = pd.DataFrame(rows)
    leaderboard = summarize_model_leaderboard(predictions)
    return predictions, leaderboard


def summarize_model_leaderboard(predictions: pd.DataFrame) -> pd.DataFrame:
    """Rank model backtests by Sharpe while keeping forecast and trading diagnostics."""
    if predictions.empty:
        return pd.DataFrame()

    rows: list[dict[str, float | int | str]] = []
    for model_name, group in predictions.groupby("model", sort=False):
        strategy = group["strategy_return"].fillna(0)
        realized = group["realized_return"].to_numpy()
        forecast = group["forecast"].fillna(0).to_numpy()
        equity = (1 + strategy).cumprod()
        drawdown = equity / equity.cummax() - 1
        active = group["signal"] != 0
        active_returns = strategy[active]
        hits = np.sign(active_returns) > 0
        years = max(len(group) / 52, 1 / 52)
        final_equity = float(equity.iloc[-1])
        weekly_vol = strategy.std()

        rows.append(
            {
                "model": str(model_name),
                "category": str(group["category"].iloc[0]),
                "observations": int(len(group)),
                "rmse": float(np.sqrt(np.mean((realized - forecast) ** 2))),
                "mae": float(np.mean(np.abs(realized - forecast))),
                "directional_accuracy": float(
                    (np.sign(forecast) == np.sign(realized)).mean()
                ),
                "total_return": final_equity - 1,
                "annual_return": final_equity ** (1 / years) - 1 if final_equity > 0 else -1.0,
                "sharpe": 0.0
                if weekly_vol == 0 or np.isnan(weekly_vol)
                else float(strategy.mean() / weekly_vol * np.sqrt(52)),
                "max_drawdown": float(drawdown.min()),
                "hit_rate": float(hits.mean()) if len(hits) else 0.0,
                "exposure": float(active.mean()),
                "turnover": float(group["signal"].diff().abs().fillna(0).mean()),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["sharpe", "annual_return"], ascending=False)
        .reset_index(drop=True)
    )


def _baseline_predictions(
    train: pd.DataFrame,
    current: pd.DataFrame,
    target: str,
    realized: float,
) -> list[dict[str, float | int | str | None]]:
    mean_forecast = float(train[target].mean())
    return [
        _prediction_row("Zero-return", "Baseline", 0.0, None, 0, realized),
        _prediction_row(
            "Historical mean",
            "Baseline",
            mean_forecast,
            None,
            _sign_signal(mean_forecast),
            realized,
        ),
    ]


def _regression_predictions(
    train: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    target: str,
    realized: float,
) -> list[dict[str, float | int | str | None]]:
    specs: list[tuple[str, Pipeline]] = [
        (
            "Ridge",
            Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        ),
        (
            "Tuned Ridge",
            Pipeline([("scale", StandardScaler()), ("model", RidgeCV(alphas=RIDGE_ALPHAS))]),
        ),
        (
            "ElasticNet",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", ElasticNet(alpha=0.001, l1_ratio=0.25, max_iter=10000)),
                ]
            ),
        ),
    ]
    rows: list[dict[str, float | int | str | None]] = []
    for name, model in specs:
        model.fit(train[features], train[target])
        forecast = float(model.predict(current[features])[0])
        rows.append(
            _prediction_row(name, "Regression", forecast, None, _sign_signal(forecast), realized)
        )
    return rows


def _classifier_predictions(
    train: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    target: str,
    realized: float,
    probability_threshold: float,
) -> list[dict[str, float | int | str | None]]:
    train_y = (train[target] > 0).astype(int)
    return_scale = float(train[target].abs().mean())
    if train_y.nunique() < 2:
        probability_up = float(train_y.mean())
        signal = signal_from_probability(probability_up, probability_threshold)
        return [
            _prediction_row(
                name,
                category,
                _classifier_return_forecast(probability_up, return_scale),
                probability_up,
                signal,
                realized,
            )
            for name, category in [
                ("Logistic regression", "Classifier"),
                ("Calibrated logistic regression", "Classifier"),
                ("Random forest", "Tree classifier"),
                ("Gradient boosting", "Tree classifier"),
            ]
        ]

    specs: list[tuple[str, str, object]] = [
        ("Logistic regression", "Classifier", make_direction_pipeline()),
        (
            "Calibrated logistic regression",
            "Classifier",
            CalibratedClassifierCV(estimator=make_direction_pipeline(), method="sigmoid", cv=3),
        ),
        (
            "Random forest",
            "Tree classifier",
            RandomForestClassifier(
                n_estimators=120,
                min_samples_leaf=8,
                random_state=7,
                class_weight="balanced_subsample",
            ),
        ),
        (
            "Gradient boosting",
            "Tree classifier",
            GradientBoostingClassifier(
                n_estimators=80,
                learning_rate=0.05,
                max_depth=2,
                random_state=7,
            ),
        ),
    ]
    rows: list[dict[str, float | int | str | None]] = []
    for name, category, model in specs:
        model.fit(train[features], train_y)
        probability_up = float(model.predict_proba(current[features])[0, 1])
        signal = signal_from_probability(probability_up, probability_threshold)
        rows.append(
            _prediction_row(
                name,
                category,
                _classifier_return_forecast(probability_up, return_scale),
                probability_up,
                signal,
                realized,
            )
        )
    return rows


def _rule_predictions(
    train: pd.DataFrame,
    current: pd.DataFrame,
    target: str,
    realized: float,
) -> list[dict[str, float | int | str | None]]:
    row = current.iloc[0]
    momentum = float(row["momentum_4w"])
    inventory_z = float(row["inventory_z_52w"])
    positioning = float(row["positioning_pct_156w"])
    drawdown = float(row["drawdown_13w"])

    trend_inventory = 0
    if momentum > 0 and inventory_z < 0.75:
        trend_inventory = 1
    elif momentum < 0 and inventory_z > -0.75:
        trend_inventory = -1

    crowded_positioning = 0
    if positioning < 0.25 and drawdown > -0.08:
        crowded_positioning = 1
    elif positioning > 0.75 and drawdown < -0.02:
        crowded_positioning = -1

    inventory_reversion = 0
    if inventory_z < -1.0:
        inventory_reversion = 1
    elif inventory_z > 1.0:
        inventory_reversion = -1

    return_scale = float(train[target].abs().mean())
    return [
        _prediction_row(
            "Rule: trend plus inventory",
            "Rule-based",
            trend_inventory * return_scale,
            None,
            trend_inventory,
            realized,
        ),
        _prediction_row(
            "Rule: positioning contrarian",
            "Rule-based",
            crowded_positioning * return_scale,
            None,
            crowded_positioning,
            realized,
        ),
        _prediction_row(
            "Rule: inventory reversion",
            "Rule-based",
            inventory_reversion * return_scale,
            None,
            inventory_reversion,
            realized,
        ),
    ]


def _prediction_row(
    model: str,
    category: str,
    forecast: float,
    probability_up: float | None,
    signal: int,
    realized: float,
) -> dict[str, float | int | str | None]:
    return {
        "model": model,
        "category": category,
        "forecast": forecast,
        "probability_up": probability_up,
        "signal": signal,
        "realized_return": realized,
        "strategy_return": signal * realized,
    }


def _sign_signal(value: float, deadband: float = 0.0) -> int:
    if value > deadband:
        return 1
    if value < -deadband:
        return -1
    return 0


def _classifier_return_forecast(probability_up: float, return_scale: float) -> float:
    return (probability_up - 0.5) * 2 * return_scale
