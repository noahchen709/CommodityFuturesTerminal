# Commodity Futures Research Terminal

A research dashboard for commodity futures that focuses on risk ranges and trade setup quality instead of simple up/down price calls.

The first milestone is a crude oil MVP:

- weekly WTI price, inventory, positioning, and macro panel
- current WTI futures curve with front-to-forward spreads
- feature engineering for returns, volatility, drawdown, inventory surprise, positioning crowding, and seasonality
- monthly and current-year seasonal return profiles
- conformal prediction interval for next-week returns
- walk-forward signal backtest
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
src/data/                         Data loaders and sample data
src/features/                     Weekly crude feature engineering
src/models/                       Conformal forecast interval
src/backtest/                     Walk-forward backtest and metrics
src/reports/                      Trading memo generator
tests/                           Smoke tests
```

## Next Milestones

1. Add production and refinery utilization to the weekly supply panel.
2. Add contract roll diagnostics and term-structure features.
3. Add source health checks and stale-data warnings in the dashboard.
4. Add scheduled refresh automation for `data/processed/crude_weekly.csv`.
