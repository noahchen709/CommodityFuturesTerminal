from src.backtest.engine import run_walk_forward_backtest, summarize_backtest
from src.data.dataset import load_crude_research_dataset
from src.features.crude_features import add_crude_features, feature_columns
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

    backtest = run_walk_forward_backtest(features, cols)
    metrics = summarize_backtest(backtest)
    assert {"sharpe", "max_drawdown", "hit_rate", "turnover"}.issubset(metrics)

    memo = generate_crude_memo(features.iloc[-1], forecast, metrics)
    assert "WTI setup" in memo
