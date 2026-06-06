from src.backtest.engine import (
    run_model_leaderboard_backtest,
    run_walk_forward_backtest,
    summarize_backtest,
)
from src.data.curves import make_wti_contract_table
from src.data.dataset import load_crude_research_dataset
from src.features.crude_features import (
    add_crude_features,
    feature_columns,
    monthly_seasonality,
    seasonal_cumulative_profile,
)
from src.models.conformal import train_conformal_forecaster
from src.reports.memo import generate_crude_memo


def test_crude_mvp_pipeline_runs() -> None:
    raw = load_crude_research_dataset()
    features = add_crude_features(raw)
    cols = feature_columns()

    assert len(features) > 100
    assert set(cols).issubset(features.columns)

    _, _, forecast = train_conformal_forecaster(features, cols)
    assert forecast.lower < forecast.point < forecast.upper
    assert forecast.probability_up is not None
    assert 0 <= forecast.probability_up <= 1
    assert forecast.signal in {-1, 0, 1}

    backtest = run_walk_forward_backtest(features, cols)
    metrics = summarize_backtest(backtest)
    assert {"sharpe", "max_drawdown", "hit_rate", "turnover"}.issubset(metrics)
    assert "probability_up" in backtest.columns

    model_bt, leaderboard = run_model_leaderboard_backtest(features, cols)
    expected_models = {
        "Zero-return",
        "Historical mean",
        "Ridge",
        "Tuned Ridge",
        "ElasticNet",
        "Logistic regression",
        "Calibrated logistic regression",
        "Random forest",
        "Gradient boosting",
        "Rule: trend plus inventory",
        "Rule: positioning contrarian",
        "Rule: inventory reversion",
    }
    assert set(leaderboard["model"]) == expected_models
    assert set(expected_models).issubset(model_bt["model"])
    assert {
        "rmse",
        "mae",
        "directional_accuracy",
        "sharpe",
        "max_drawdown",
        "exposure",
    }.issubset(leaderboard.columns)

    memo = generate_crude_memo(features.iloc[-1], forecast, metrics)
    assert "WTI setup" in memo


def test_curve_and_seasonality_helpers_run() -> None:
    features = add_crude_features(load_crude_research_dataset())

    contracts = make_wti_contract_table(months=6)
    assert len(contracts) == 6
    assert {"symbol", "contract_month", "months_forward"}.issubset(contracts.columns)

    monthly = monthly_seasonality(features)
    assert len(monthly) == 12
    assert {"avg_return", "positive_rate", "observations"}.issubset(monthly.columns)

    seasonal = seasonal_cumulative_profile(features)
    assert {"seasonal_avg", "seasonal_low", "seasonal_high", "current_year"}.issubset(
        seasonal.columns
    )
