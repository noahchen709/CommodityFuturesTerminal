from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from urllib.parse import urlencode

import pandas as pd

from src.data.prices import fetch_wti_yfinance


CFTC_DISAGGREGATED_FUTURES_ONLY_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.csv"
CFTC_WTI_CONTRACT_MARKET_NAME = "WTI-PHYSICAL"
EIA_WTI_INVENTORY_XLS_URL = "https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls"
DOLLAR_INDEX_FRED_SERIES = "DTWEXBGS"
DEFAULT_START_DATE = pd.Timestamp("2006-06-01")
HTTP_TIMEOUT_SECONDS = 8
HTTP_RETRY_ATTEMPTS = 1
HTTP_RETRY_DELAY_SECONDS = 0.5


def build_real_crude_weekly_dataset(
    years: int | None = None,
    fallback_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a weekly crude research panel from live public data sources."""
    end = pd.Timestamp.today().normalize()
    start = DEFAULT_START_DATE if years is None else end - pd.DateOffset(years=years + 1)
    cached_dollar = recent_cached_source_frame(
        fallback_panel,
        ["dollar_index"],
        start,
        end,
        max_age_days=10,
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        prices_future = executor.submit(
            fetch_wti_yfinance,
            period="max" if years is None else f"{years + 1}y",
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        inventory_future = executor.submit(fetch_eia_wti_inventory_change, start=start, end=end)
        dollar_future = None
        if cached_dollar is None:
            dollar_future = executor.submit(
                fetch_fred_weekly_series,
                DOLLAR_INDEX_FRED_SERIES,
                "dollar_index",
                start,
                end,
            )
        positioning_future = executor.submit(fetch_cftc_wti_managed_money_net, start=start)

        issues = []
        prices, issue = future_result_or_cached(
            "Yahoo Finance WTI prices",
            prices_future,
            fallback_panel,
            ["settle"],
            start,
            end,
        )
        if issue:
            issues.append(issue)
        inventory, issue = future_result_or_cached(
            "EIA crude inventories",
            inventory_future,
            fallback_panel,
            ["inventory_change_mmbbl"],
            start,
            end,
        )
        if issue:
            issues.append(issue)
        if cached_dollar is not None:
            dollar = cached_dollar
        else:
            dollar, issue = future_result_or_cached(
                "FRED dollar index",
                dollar_future,
                fallback_panel,
                ["dollar_index"],
                start,
                end,
            )
            if issue:
                issues.append(issue)
        positioning, issue = future_result_or_cached(
            "CFTC positioning",
            positioning_future,
            fallback_panel,
            ["managed_money_net_k"],
            start,
            end,
        )
        if issue:
            issues.append(issue)

    panel = merge_weekly_panel(
        prices=prices[prices["date"].between(start, end)].copy(),
        inventory=inventory[["date", "inventory_change_mmbbl"]],
        positioning=positioning,
        dollar=dollar,
    )
    if years is not None:
        panel = panel.tail(years * 52)
    panel.attrs["refresh_issues"] = issues
    return panel.reset_index(drop=True)


def refresh_real_crude_weekly_dataset(
    output_path: str | Path = "data/processed/crude_weekly.csv",
    years: int | None = None,
) -> pd.DataFrame:
    """Fetch live inputs, persist the processed research panel, and return it."""
    output = Path(output_path)
    cached = read_cached_panel(output) if years is None else None
    refresh_years = years or incremental_refresh_years(cached)
    panel = build_real_crude_weekly_dataset(years=refresh_years, fallback_panel=cached)
    issues = panel.attrs.get("refresh_issues", [])
    if cached is not None:
        panel = (
            pd.concat([cached, panel], ignore_index=True)
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
        panel.attrs["refresh_issues"] = issues
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
    raw = read_csv_url(url, parse_dates=["observation_date"], na_values=["."])
    series = raw.set_index("observation_date")[series_id].astype(float)
    weekly = series.resample("W-FRI").last().ffill()
    return weekly.rename(column).reset_index().rename(columns={"observation_date": "date"})


def fetch_eia_wti_inventory_change(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch EIA weekly U.S. crude stocks excluding SPR and return changes in mmbbl."""
    raw = pd.read_excel(BytesIO(read_url_bytes(EIA_WTI_INVENTORY_XLS_URL)), sheet_name=1)
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
    raw = read_csv_url(url)

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


def read_cached_panel(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])


def incremental_refresh_years(cached: pd.DataFrame | None) -> int | None:
    if cached is None or cached.empty or "date" not in cached.columns:
        return None

    latest_cached = pd.to_datetime(cached["date"]).max()
    days_stale = max((pd.Timestamp.today().normalize() - latest_cached).days, 0)
    return max(1, int(days_stale / 365) + 1)


def future_result_or_cached(
    source: str,
    future,
    fallback_panel: pd.DataFrame | None,
    columns: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, str | None]:
    try:
        return future.result(), None
    except Exception as exc:
        fallback = cached_source_frame(fallback_panel, columns, start, end)
        issue = f"{source} refresh failed; reused cached {', '.join(columns)}. ({exc})"
        if fallback is None:
            raise RuntimeError(f"{source} refresh failed and no cached fallback is available: {exc}") from exc
        return fallback, issue


def cached_source_frame(
    fallback_panel: pd.DataFrame | None,
    columns: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame | None:
    if fallback_panel is None or fallback_panel.empty:
        return None

    required = ["date", *columns]
    if any(column not in fallback_panel.columns for column in required):
        return None

    out = fallback_panel[required].copy()
    out = normalize_dates(out).dropna(subset=required).sort_values("date")
    out = out[out["date"].between(start - pd.Timedelta(days=14), end)]
    if out.empty:
        return None
    return out.reset_index(drop=True)


def recent_cached_source_frame(
    fallback_panel: pd.DataFrame | None,
    columns: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_age_days: int,
) -> pd.DataFrame | None:
    frame = cached_source_frame(fallback_panel, columns, start, end)
    if frame is None:
        return None
    if frame["date"].max() < end - pd.Timedelta(days=max_age_days):
        return None
    return frame


def read_csv_url(url: str, **kwargs: object) -> pd.DataFrame:
    return pd.read_csv(BytesIO(read_url_bytes(url)), **kwargs)


def read_url_bytes(url: str) -> bytes:
    for attempt in range(HTTP_RETRY_ATTEMPTS + 1):
        try:
            with urlopen(url, timeout=HTTP_TIMEOUT_SECONDS) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code not in {502, 503, 504} or attempt == HTTP_RETRY_ATTEMPTS:
                raise
        except (TimeoutError, URLError):
            if attempt == HTTP_RETRY_ATTEMPTS:
                raise
        time.sleep(HTTP_RETRY_DELAY_SECONDS * (attempt + 1))

    raise RuntimeError("Unreachable URL retry state.")
