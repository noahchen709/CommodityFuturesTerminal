from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.data.real import refresh_real_crude_weekly_dataset
from src.data.sample import make_sample_crude_weekly_data


DEFAULT_PROCESSED_PATH = Path("data/processed/crude_weekly.csv")
REQUIRED_COLUMNS = {
    "date",
    "settle",
    "inventory_change_mmbbl",
    "managed_money_net_k",
    "dollar_index",
}


def load_crude_research_dataset(
    path: str | None = None,
    source: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load the weekly crude research panel from sample, CSV cache, or live sources."""
    source = source or os.getenv("CRUDE_DATA_SOURCE", "sample")

    if path is not None:
        df = pd.read_csv(path, parse_dates=["date"])
    elif source == "sample":
        df = make_sample_crude_weekly_data()
    elif source == "csv":
        df = pd.read_csv(DEFAULT_PROCESSED_PATH, parse_dates=["date"])
    elif source == "live":
        if refresh or not DEFAULT_PROCESSED_PATH.exists():
            df = refresh_real_crude_weekly_dataset(DEFAULT_PROCESSED_PATH)
        else:
            df = pd.read_csv(DEFAULT_PROCESSED_PATH, parse_dates=["date"])
    else:
        raise ValueError("source must be one of: sample, csv, live")

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    return df.sort_values("date").dropna(subset=["date", "settle"]).reset_index(drop=True)
