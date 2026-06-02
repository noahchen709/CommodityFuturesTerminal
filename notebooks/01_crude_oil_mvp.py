from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import run_walk_forward_backtest, summarize_backtest
from src.data.dataset import load_crude_research_dataset
from src.features.crude_features import add_crude_features, feature_columns
from src.models.conformal import train_conformal_forecaster
from src.reports.memo import generate_crude_memo


def main() -> None:
    raw = load_crude_research_dataset()
    features = add_crude_features(raw)
    cols = feature_columns()

    _, _, forecast = train_conformal_forecaster(features, cols)
    backtest = run_walk_forward_backtest(features, cols)
    metrics = summarize_backtest(backtest)
    memo = generate_crude_memo(features.iloc[-1], forecast, metrics)

    print("Dataset rows:", len(features))
    print("Latest forecast:", forecast)
    print("Backtest metrics:", metrics)
    print()
    print(memo)


if __name__ == "__main__":
    main()
