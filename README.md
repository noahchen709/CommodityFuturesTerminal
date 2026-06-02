# Commodity Futures Research Terminal

A research dashboard for commodity futures that focuses on risk ranges and trade setup quality instead of simple up/down price calls.

The first milestone is a crude oil MVP:

- weekly WTI price, inventory, positioning, and macro panel
- feature engineering for returns, volatility, drawdown, inventory surprise, positioning crowding, and seasonality
- conformal prediction interval for next-week returns
- walk-forward signal backtest
- compact trading memo generator
- Streamlit dashboard

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python notebooks/01_crude_oil_mvp.py
streamlit run app/dashboard.py
```

The current MVP runs offline with generated sample data. Replace `load_crude_research_dataset()` with a processed CSV once live EIA, CFTC, and FRED ingestion is added.

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

1. Add live WTI price ingestion through `yfinance`.
2. Add EIA weekly crude inventory and production ingestion.
3. Add CFTC disaggregated COT positioning for crude oil.
4. Add FRED macro factors.
5. Persist a weekly processed research panel to `data/processed/crude_weekly.csv`.
