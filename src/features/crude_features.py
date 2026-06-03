from __future__ import annotations

import numpy as np
import pandas as pd


def add_crude_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build weekly crude oil research features and next-week target."""
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["return_1w"] = out["settle"].pct_change()
    out["target_return_1w"] = out["return_1w"].shift(-1)
    out["momentum_4w"] = out["settle"].pct_change(4)
    out["momentum_12w"] = out["settle"].pct_change(12)
    out["vol_8w"] = out["return_1w"].rolling(8).std() * np.sqrt(52)
    out["drawdown_13w"] = out["settle"] / out["settle"].rolling(13).max() - 1
    out["inventory_z_52w"] = zscore(out["inventory_change_mmbbl"], 52)
    out["positioning_pct_156w"] = rolling_percentile(out["managed_money_net_k"], 156)
    out["dollar_return_4w"] = out["dollar_index"].pct_change(4)
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.quarter
    return out.dropna(subset=feature_columns()).reset_index(drop=True)


def feature_columns() -> list[str]:
    return [
        "return_1w",
        "momentum_4w",
        "momentum_12w",
        "vol_8w",
        "drawdown_13w",
        "inventory_z_52w",
        "positioning_pct_156w",
        "dollar_return_4w",
        "month",
        "quarter",
    ]


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(8, window // 4)).mean()
    std = series.rolling(window, min_periods=max(8, window // 4)).std()
    return (series - mean) / std.replace(0, np.nan)


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    min_periods = max(12, window // 4)

    def pct_rank(values: np.ndarray) -> float:
        latest = values[-1]
        return float((values <= latest).mean())

    return series.rolling(window, min_periods=min_periods).apply(pct_rank, raw=True)


def monthly_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize historical WTI weekly returns by calendar month."""
    data = df.dropna(subset=["return_1w"]).copy()
    grouped = data.groupby(data["date"].dt.month)["return_1w"]
    return pd.DataFrame(
        {
            "month": range(1, 13),
            "avg_return": grouped.mean().reindex(range(1, 13)).to_numpy(),
            "positive_rate": grouped.apply(lambda x: (x > 0).mean()).reindex(range(1, 13)).to_numpy(),
            "observations": grouped.count().reindex(range(1, 13)).fillna(0).astype(int).to_numpy(),
        }
    )


def seasonal_cumulative_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Compare current-year cumulative WTI returns with historical seasonal bands."""
    data = df.dropna(subset=["return_1w"]).copy()
    data["year"] = data["date"].dt.year
    data["week"] = data["date"].dt.isocalendar().week.astype(int)
    data["cumulative_return"] = data.groupby("year")["return_1w"].transform(
        lambda x: (1 + x).cumprod() - 1
    )

    latest_year = int(data["year"].max())
    historical = data[data["year"] < latest_year]
    current = data[data["year"] == latest_year][["week", "cumulative_return"]].rename(
        columns={"cumulative_return": "current_year"}
    )

    bands = historical.groupby("week")["cumulative_return"].agg(
        seasonal_avg="mean",
        seasonal_low=lambda x: x.quantile(0.2),
        seasonal_high=lambda x: x.quantile(0.8),
    )
    return (
        bands.reset_index()
        .merge(current, on="week", how="left")
        .sort_values("week")
        .reset_index(drop=True)
    )
