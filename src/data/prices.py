from __future__ import annotations

import pandas as pd


def load_wti_prices_from_csv(path: str) -> pd.DataFrame:
    """Load WTI futures prices from a CSV with date and settle/close columns."""
    df = pd.read_csv(path, parse_dates=["date"])
    close_col = "settle" if "settle" in df.columns else "close"
    return (
        df.rename(columns={close_col: "settle"})[["date", "settle"]]
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )


def fetch_wti_yfinance(symbol: str = "CL=F", period: str = "7y") -> pd.DataFrame:
    """Fetch daily WTI futures prices and resample to weekly Friday closes."""
    import yfinance as yf

    data = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise RuntimeError(f"No price data returned for {symbol}.")

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    weekly = close.resample("W-FRI").last().dropna()
    return weekly.rename("settle").reset_index().rename(columns={"Date": "date"})
