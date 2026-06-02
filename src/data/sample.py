from __future__ import annotations

import numpy as np
import pandas as pd


def make_sample_crude_weekly_data(seed: int = 7, periods: int = 260) -> pd.DataFrame:
    """Create a realistic offline weekly crude dataset for local development."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-04", periods=periods, freq="W-FRI")

    seasonal = np.sin(np.linspace(0, 10 * np.pi, periods))
    macro_shock = rng.normal(0, 0.35, periods).cumsum()
    inventory_change = rng.normal(0, 4.5, periods) - 1.5 * seasonal
    managed_money_net = 180 + 35 * seasonal + rng.normal(0, 18, periods)
    dollar_index = 100 + macro_shock + rng.normal(0, 0.25, periods)

    returns = (
        0.0015
        - 0.0018 * inventory_change
        + 0.000025 * (managed_money_net - managed_money_net.mean())
        - 0.003 * np.diff(np.r_[dollar_index[0], dollar_index])
        + rng.normal(0, 0.035, periods)
    )
    price = 65 * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {
            "date": dates,
            "settle": price,
            "inventory_change_mmbbl": inventory_change,
            "managed_money_net_k": managed_money_net,
            "dollar_index": dollar_index,
        }
    )
