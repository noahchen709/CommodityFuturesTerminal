from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from src.data.prices import fetch_wti_yfinance


CFTC_DISAGGREGATED_FUTURES_ONLY_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.csv"
CFTC_WTI_CONTRACT_MARKET_NAME = "WTI-PHYSICAL"
EIA_WTI_INVENTORY_XLS_URL = "https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls"
DOLLAR_INDEX_FRED_SERIES = "DTWEXBGS"
DEFAULT_START_DATE = pd.Timestamp("2006-06-01")


def build_real_crude_weekly_dataset(years: int | None = None) -> pd.DataFrame:
    """Build a weekly crude research panel from live public data sources."""
    end = pd.Timestamp.today().normalize()
    start = DEFAULT_START_DATE if years is None else end - pd.DateOffset(years=years + 1)

    prices = fetch_wti_yfinance(period="max" if years is None else f"{years + 1}y")
    prices = prices[prices["date"].between(start, end)].copy()

    inventory = fetch_eia_wti_inventory_change(start=start, end=end)

    dollar = fetch_fred_weekly_series(
        DOLLAR_INDEX_FRED_SERIES,
        "dollar_index",
        start=start,
        end=end,
    )
    positioning = fetch_cftc_wti_managed_money_net(start=start)

    panel = merge_weekly_panel(
        prices=prices,
        inventory=inventory[["date", "inventory_change_mmbbl"]],
        positioning=positioning,
        dollar=dollar,
    )
    if years is not None:
        panel = panel.tail(years * 52)
    return panel.reset_index(drop=True)


def refresh_real_crude_weekly_dataset(
    output_path: str | Path = "data/processed/crude_weekly.csv",
    years: int | None = None,
) -> pd.DataFrame:
    """Fetch live inputs, persist the processed research panel, and return it."""
    panel = build_real_crude_weekly_dataset(years=years)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output, index=False)
    return panel


def fetch_fred_weekly_series(
    series_id: str,
    column: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch a FRED series and align it to Friday weekly observations."""
    params = urlencode(
        {
            "id": series_id,
            "cosd": start.strftime("%Y-%m-%d"),
            "coed": end.strftime("%Y-%m-%d"),
        }
    )
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{params}"
    raw = pd.read_csv(url, parse_dates=["observation_date"], na_values=["."])
    series = raw.set_index("observation_date")[series_id].astype(float)
    weekly = series.resample("W-FRI").last().ffill()
    return weekly.rename(column).reset_index().rename(columns={"observation_date": "date"})


def fetch_eia_wti_inventory_change(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch EIA weekly U.S. crude stocks excluding SPR and return changes in mmbbl."""
    raw = pd.read_excel(EIA_WTI_INVENTORY_XLS_URL, sheet_name=1)
    date_col = raw.columns[0]
    value_col = raw.columns[1]
    rows = raw.iloc[2:].copy()
    stocks = pd.DataFrame(
        {
            "date": pd.to_datetime(rows[date_col], errors="coerce"),
            "crude_stocks_kbbl": pd.to_numeric(rows[value_col], errors="coerce"),
        }
    ).dropna()
    stocks = stocks[stocks["date"].between(start, end)].sort_values("date")
    stocks["inventory_change_mmbbl"] = stocks["crude_stocks_kbbl"].diff() / 1000
    return stocks[["date", "inventory_change_mmbbl"]].dropna().reset_index(drop=True)


def fetch_cftc_wti_managed_money_net(start: pd.Timestamp) -> pd.DataFrame:
    """Fetch WTI managed-money net futures positioning from CFTC Socrata."""
    query = f"""
        SELECT
            report_date_as_yyyy_mm_dd,
            market_and_exchange_names,
            m_money_positions_long_all,
            m_money_positions_short_all
        WHERE contract_market_name = '{CFTC_WTI_CONTRACT_MARKET_NAME}'
        AND report_date_as_yyyy_mm_dd >= '{start.strftime("%Y-%m-%d")}T00:00:00'
        ORDER BY report_date_as_yyyy_mm_dd
        LIMIT 50000
    """
    url = f"{CFTC_DISAGGREGATED_FUTURES_ONLY_URL}?{urlencode({'$query': ' '.join(query.split())})}"
    raw = pd.read_csv(url)

    if raw.empty:
        raise RuntimeError("No CFTC WTI managed-money positioning rows returned.")

    long_col = "m_money_positions_long_all"
    short_col = "m_money_positions_short_all"
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["report_date_as_yyyy_mm_dd"]).dt.tz_localize(None),
            "managed_money_net_k": (
                pd.to_numeric(raw[long_col], errors="coerce")
                - pd.to_numeric(raw[short_col], errors="coerce")
            )
            / 1000,
        }
    )
    return out.dropna().sort_values("date").reset_index(drop=True)


def merge_weekly_panel(
    prices: pd.DataFrame,
    inventory: pd.DataFrame,
    positioning: pd.DataFrame,
    dollar: pd.DataFrame,
) -> pd.DataFrame:
    """Align weekly data sources onto price Fridays with backward as-of joins."""
    panel = normalize_dates(prices[["date", "settle"]]).sort_values("date")
    for frame in (inventory, positioning, dollar):
        panel = pd.merge_asof(
            panel,
            normalize_dates(frame).sort_values("date"),
            on="date",
            direction="backward",
            tolerance=pd.Timedelta(days=10),
        )

    required = [
        "settle",
        "inventory_change_mmbbl",
        "managed_money_net_k",
        "dollar_index",
    ]
    return panel.dropna(subset=required).reset_index(drop=True)


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).astype("datetime64[ns]")
    return out
