# Commodity Futures Research Terminal

A research terminal for commodity futures that focuses on professional data workflows, honest walk-forward validation, uncertainty-aware forecasting, and risk-managed trading research.

The first milestone is a crude oil MVP:

- weekly WTI price, inventory, positioning, and macro panel
- current WTI futures curve with front-to-forward spreads
- feature engineering for returns, volatility, drawdown, inventory surprise, positioning crowding, and seasonality
- monthly and current-year seasonal return profiles
- conformal prediction interval for next-week returns
- walk-forward signal backtest
- model leaderboard for baselines, linear models, classifiers, tree models, and rule-based strategies
- compact trading memo generator
- Streamlit dashboard

## Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/dashboard.py
```

Open the Streamlit URL shown in the terminal, usually `http://localhost:8501`.

The app uses real data by default. On first run, or when you click **Refresh live data**, it builds/updates `data/processed/crude_weekly.csv` from public sources. It does not use generated sample data for the dashboard.

To refresh the real weekly panel from the command line:

```bash
python -c "from src.data.real import refresh_real_crude_weekly_dataset; refresh_real_crude_weekly_dataset()"
```

Live sources: Yahoo Finance WTI futures prices and curve contracts, EIA crude inventories, CFTC WTI managed-money positioning, and FRED broad dollar index. The joined weekly panel currently starts in 2006.

## Project Structure

```text
app/dashboard.py                 Streamlit dashboard
notebooks/01_crude_oil_mvp.py    Runnable crude MVP script
ROADMAP.md                       Research, profitability roadmap
src/data/                         Data loaders and sample data
src/features/                     Weekly crude feature engineering
src/models/                       Conformal forecast interval
src/backtest/                     Walk-forward backtest and metrics
src/reports/                      Trading memo generator
tests/                           Smoke tests
```

## Next Milestones

See [ROADMAP.md](ROADMAP.md) for the full research plan. The near-term priorities are:

1. Add transaction costs, slippage, and cost-adjusted backtests.
2. Add transaction-cost-aware leaderboard rankings and richer model diagnostics.
3. Add probability calibration diagnostics and reliability curves.
4. Add futures curve, roll yield, and term-structure features.
5. Add source health checks and stale-data warnings in the dashboard.
