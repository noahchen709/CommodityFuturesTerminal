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
    return out.dropna().reset_index(drop=True)


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
