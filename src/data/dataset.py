from __future__ import annotations

import pandas as pd

from src.data.sample import make_sample_crude_weekly_data


REQUIRED_COLUMNS = {
    "date",
    "settle",
    "inventory_change_mmbbl",
    "managed_money_net_k",
    "dollar_index",
}


def load_crude_research_dataset(path: str | None = None) -> pd.DataFrame:
    """Load the weekly crude research panel, falling back to sample data."""
    if path is None:
        df = make_sample_crude_weekly_data()
    else:
        df = pd.read_csv(path, parse_dates=["date"])

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    return df.sort_values("date").dropna(subset=["date", "settle"]).reset_index(drop=True)
