from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


MONTH_CODES = {
    1: "F",
    2: "G",
    3: "H",
    4: "J",
    5: "K",
    6: "M",
    7: "N",
    8: "Q",
    9: "U",
    10: "V",
    11: "X",
    12: "Z",
}


@dataclass(frozen=True)
class CurveSnapshot:
    curve: pd.DataFrame
    front_settle: float
    six_month_spread: float
    twelve_month_spread: float
    source: str = "live"


def fetch_wti_futures_curve(months: int = 18, lookback: str = "10d") -> CurveSnapshot:
    """Fetch the current WTI futures curve from Yahoo Finance contract symbols."""
    import yfinance as yf

    contracts = make_wti_contract_table(months=months)
    data = yf.download(
        contracts["symbol"].tolist(),
        period=lookback,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    rows: list[dict[str, object]] = []
    for contract in contracts.itertuples(index=False):
        close = contract_close(data, contract.symbol).dropna()
        if close.empty:
            continue
        rows.append(
            {
                "symbol": contract.symbol,
                "contract_month": contract.contract_month,
                "months_forward": contract.months_forward,
                "settle": float(close.iloc[-1]),
                "date": pd.Timestamp(close.index[-1]).tz_localize(None),
            }
        )

    curve = pd.DataFrame(rows).sort_values("contract_month").reset_index(drop=True)
    if curve.empty:
        raise RuntimeError("No WTI futures curve contracts returned from Yahoo Finance.")

    front = float(curve["settle"].iloc[0])
    six = spread_to_month(curve, 6)
    twelve = spread_to_month(curve, 12)
    return CurveSnapshot(
        curve=curve,
        front_settle=front,
        six_month_spread=six,
        twelve_month_spread=twelve,
        source="live",
    )


def make_wti_contract_table(months: int = 18, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Generate Yahoo Finance WTI NYMEX contract symbols around the current curve."""
    start = (as_of or pd.Timestamp.today()).normalize().replace(day=1) + pd.offsets.MonthBegin(1)
    contract_months = pd.date_range(start, periods=months, freq="MS")
    rows = []
    for idx, contract_month in enumerate(contract_months):
        year_suffix = contract_month.strftime("%y")
        code = MONTH_CODES[int(contract_month.month)]
        rows.append(
            {
                "symbol": f"CL{code}{year_suffix}.NYM",
                "contract_month": contract_month,
                "months_forward": idx,
            }
        )
    return pd.DataFrame(rows)


def contract_close(data: pd.DataFrame, symbol: str) -> pd.Series:
    if isinstance(data.columns, pd.MultiIndex):
        if symbol not in data.columns.get_level_values(0):
            return pd.Series(dtype=float)
        close = data[symbol]["Close"]
    else:
        close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close


def spread_to_month(curve: pd.DataFrame, month: int) -> float:
    if len(curve) <= month:
        return 0.0
    return float(curve["settle"].iloc[0] - curve["settle"].iloc[month])
