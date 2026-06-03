from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import run_walk_forward_backtest, summarize_backtest
from src.data.curves import CurveSnapshot, fetch_wti_futures_curve, make_wti_contract_table
from src.data.dataset import load_crude_research_dataset
from src.features import crude_features
from src.models.conformal import ForecastResult, train_conformal_forecaster
from src.reports.memo import generate_crude_memo


BACKTEST_TRAIN_WINDOW = 156
BACKTEST_CACHE_VERSION = 2
MODEL_OUTPUT_DIR = Path("data/processed")
BACKTEST_OUTPUT_PATH = MODEL_OUTPUT_DIR / "backtest_walk_forward.csv"
FORECAST_OUTPUT_PATH = MODEL_OUTPUT_DIR / "latest_forecast.json"

TIME_RANGES = {
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "1Y": pd.DateOffset(years=1),
    "5Y": pd.DateOffset(years=5),
    "10Y": pd.DateOffset(years=10),
    "All": None,
}


@st.cache_data(show_spinner=False)
def load_dashboard_data(refresh: bool = False) -> pd.DataFrame:
    return load_crude_research_dataset(source="live", refresh=refresh)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_curve_data(refresh: bool = False) -> CurveSnapshot:
    if refresh:
        try:
            return fetch_wti_futures_curve()
        except Exception:
            pass
    return build_cached_curve_snapshot(load_crude_research_dataset(source="csv"))


def build_cached_curve_snapshot(raw: pd.DataFrame, months: int = 18) -> CurveSnapshot:
    latest = raw.sort_values("date").iloc[-1]
    curve = make_wti_contract_table(months=months, as_of=pd.Timestamp(latest["date"]))
    curve["settle"] = float(latest["settle"])
    curve["date"] = pd.Timestamp(latest["date"])
    return CurveSnapshot(
        curve=curve,
        front_settle=float(latest["settle"]),
        six_month_spread=0.0,
        twelve_month_spread=0.0,
    )


def build_monthly_seasonality(df: pd.DataFrame) -> pd.DataFrame:
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


def build_seasonal_cumulative_profile(df: pd.DataFrame) -> pd.DataFrame:
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


@st.cache_data(show_spinner=False)
def build_research_outputs(
    raw: pd.DataFrame,
    recompute_models: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, ForecastResult, pd.DataFrame]:
    features = crude_features.add_crude_features(raw)
    monthly = build_monthly_seasonality(features)
    seasonal = build_seasonal_cumulative_profile(features)
    if not recompute_models:
        cached = load_model_outputs()
        if cached is not None:
            residual_q, forecast, bt = cached
            return features, monthly, seasonal, residual_q, forecast, bt

    cols = crude_features.feature_columns()
    _, residual_q, forecast = train_conformal_forecaster(features, cols)
    bt = run_walk_forward_backtest(features, cols, train_window=BACKTEST_TRAIN_WINDOW)
    save_model_outputs(residual_q, forecast, bt, features["date"].max())
    return features, monthly, seasonal, residual_q, forecast, bt


def load_model_outputs() -> tuple[float, ForecastResult, pd.DataFrame] | None:
    if not BACKTEST_OUTPUT_PATH.exists() or not FORECAST_OUTPUT_PATH.exists():
        return None

    with FORECAST_OUTPUT_PATH.open() as handle:
        payload = json.load(handle)
    if int(payload.get("backtest_cache_version", 0)) != BACKTEST_CACHE_VERSION:
        return None

    forecast = ForecastResult(
        point=float(payload["point"]),
        lower=float(payload["lower"]),
        upper=float(payload["upper"]),
        interval_width=float(payload["interval_width"]),
        alpha=float(payload["alpha"]),
    )
    bt = pd.read_csv(BACKTEST_OUTPUT_PATH, parse_dates=["date"])
    return float(payload["residual_q"]), forecast, bt


def save_model_outputs(
    residual_q: float,
    forecast: ForecastResult,
    bt: pd.DataFrame,
    latest_date: pd.Timestamp,
) -> None:
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bt.to_csv(BACKTEST_OUTPUT_PATH, index=False)
    payload = {
        "point": forecast.point,
        "lower": forecast.lower,
        "upper": forecast.upper,
        "interval_width": forecast.interval_width,
        "alpha": forecast.alpha,
        "residual_q": residual_q,
        "latest_data_date": latest_date.strftime("%Y-%m-%d"),
        "backtest_cache_version": BACKTEST_CACHE_VERSION,
        "backtest_scope": "all_post_training",
        "backtest_train_window": BACKTEST_TRAIN_WINDOW,
    }
    FORECAST_OUTPUT_PATH.write_text(json.dumps(payload, indent=2))


def filter_time_range(df: pd.DataFrame, label: str) -> pd.DataFrame:
    offset = TIME_RANGES[label]
    if offset is None or df.empty:
        return df

    latest_date = df["date"].max()
    return df[df["date"] >= latest_date - offset].reset_index(drop=True)


def style_time_series(
    fig: go.Figure,
    title: str,
    y_title: str,
    height: int,
    y_tickformat: str | None = None,
) -> None:
    fig.update_xaxes(title_text="Date", type="date", showgrid=False)
    fig.update_yaxes(title_text=y_title, tickformat=y_tickformat, zeroline=False)
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left"),
        height=height,
        margin=dict(l=20, r=20, t=55, b=45),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )


st.set_page_config(page_title="Commodity Futures Research Terminal", layout="wide")

title_col, refresh_col, model_col = st.columns([3.6, 1, 1.2])
title_col.title("Commodity Futures Research Terminal")
refresh = refresh_col.button("Refresh live data", width="stretch")
recompute_models = model_col.button("Run models", width="stretch")
st.caption("Crude oil MVP: supply-demand, positioning, volatility, conformal range, and trade memo.")

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        overflow: visible;
        text-overflow: clip;
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if refresh:
    load_dashboard_data.clear()
    load_curve_data.clear()
if refresh or recompute_models:
    build_research_outputs.clear()

raw = load_dashboard_data(refresh=refresh)
curve_snapshot = load_curve_data(refresh=refresh)
features, monthly, seasonal, residual_q, forecast, bt = build_research_outputs(
    raw,
    recompute_models=refresh or recompute_models,
)
latest = features.iloc[-1]

time_range = st.segmented_control(
    "Market window",
    list(TIME_RANGES),
    default="1Y",
    label_visibility="collapsed",
)
visible_features = filter_time_range(features, time_range)
visible_bt = filter_time_range(bt, time_range)
metrics = summarize_backtest(visible_bt if not visible_bt.empty else bt)

metric_cols = st.columns([1, 1, 1.75, 1, 1.35, 1.2], gap="large")
metric_cols[0].metric("WTI settle", f"${latest['settle']:.2f}")
metric_cols[1].metric("Forecast", f"{forecast.point:.2%}")
metric_cols[2].metric("Range", f"{forecast.lower:.2%} to {forecast.upper:.2%}")
metric_cols[3].metric("Vol 8w", f"{latest['vol_8w']:.1%}")
metric_cols[4].metric("6M curve spread", f"${curve_snapshot.six_month_spread:.2f}")
metric_cols[5].metric("Backtest Sharpe", f"{metrics['sharpe']:.2f}")

st.caption(
    f"Real data through {latest['date']:%Y-%m-%d} | "
    f"{len(visible_features):,} of {len(features):,} feature rows shown"
)

tab_market, tab_curve, tab_seasonality, tab_forecast, tab_backtest, tab_memo = st.tabs(
    ["Market", "Futures Curve", "Seasonality", "Forecast Range", "Backtest", "Trading Memo"]
)

with tab_market:
    price_fig = go.Figure()
    price_fig.add_trace(
        go.Scatter(
            x=visible_features["date"],
            y=visible_features["settle"],
            name="WTI settle",
            hovertemplate="$%{y:.2f}<extra></extra>",
        )
    )
    style_time_series(price_fig, "WTI Futures Settlement Price", "Dollars per barrel", 420)
    st.plotly_chart(price_fig, width="stretch")

    left, right = st.columns(2)
    inventory_fig = go.Figure()
    inventory_fig.add_hline(y=0, line_dash="dot", line_color="gray")
    inventory_fig.add_hline(y=1, line_dash="dot", line_color="#d62728")
    inventory_fig.add_hline(y=-1, line_dash="dot", line_color="#2ca02c")
    inventory_fig.add_trace(
        go.Scatter(
            x=visible_features["date"],
            y=visible_features["inventory_z_52w"],
            name="Inventory change z-score",
            hovertemplate="%{y:.2f} z<extra></extra>",
        )
    )
    style_time_series(
        inventory_fig,
        "Crude Inventory Pressure",
        "52-week z-score",
        300,
    )
    left.plotly_chart(inventory_fig, width="stretch")

    positioning_fig = go.Figure()
    positioning_fig.add_hline(y=0.8, line_dash="dot", line_color="#d62728")
    positioning_fig.add_hline(y=0.2, line_dash="dot", line_color="#2ca02c")
    positioning_fig.add_trace(
        go.Scatter(
            x=visible_features["date"],
            y=visible_features["positioning_pct_156w"],
            name="Managed-money net percentile",
            hovertemplate="%{y:.0%}<extra></extra>",
        )
    )
    style_time_series(
        positioning_fig,
        "Managed-Money Positioning",
        "156-week percentile",
        300,
        y_tickformat=".0%",
    )
    right.plotly_chart(positioning_fig, width="stretch")

with tab_curve:
    curve = curve_snapshot.curve
    curve_fig = go.Figure()
    curve_fig.add_trace(
        go.Scatter(
            x=curve["contract_month"],
            y=curve["settle"],
            mode="lines+markers",
            name="WTI curve",
            customdata=curve[["symbol", "months_forward"]],
            hovertemplate="%{customdata[0]}<br>%{x|%b %Y}<br>$%{y:.2f}<extra></extra>",
        )
    )
    curve_fig.update_xaxes(title_text="Contract month", type="date")
    curve_fig.update_yaxes(title_text="Dollars per barrel")
    curve_fig.update_layout(
        title=dict(text="Current WTI Futures Curve", x=0, xanchor="left"),
        height=430,
        margin=dict(l=20, r=20, t=55, b=45),
        hovermode="x unified",
    )
    st.plotly_chart(curve_fig, width="stretch")

    curve_cols = st.columns(3)
    curve_cols[0].metric("Front contract", f"${curve_snapshot.front_settle:.2f}")
    curve_cols[1].metric("Front less 6M", f"${curve_snapshot.six_month_spread:.2f}")
    curve_cols[2].metric("Front less 12M", f"${curve_snapshot.twelve_month_spread:.2f}")

with tab_seasonality:
    month_fig = go.Figure()
    month_fig.add_trace(
        go.Bar(
            x=monthly["month"],
            y=monthly["avg_return"],
            name="Average weekly return",
            hovertemplate="Month %{x}<br>%{y:.2%}<extra></extra>",
        )
    )
    month_fig.update_xaxes(title_text="Calendar month", dtick=1)
    month_fig.update_yaxes(title_text="Average weekly return", tickformat=".1%")
    month_fig.update_layout(
        title=dict(text="Monthly Return Seasonality", x=0, xanchor="left"),
        height=360,
        margin=dict(l=20, r=20, t=55, b=45),
    )
    st.plotly_chart(month_fig, width="stretch")

    seasonal_fig = go.Figure()
    seasonal_fig.add_trace(
        go.Scatter(
            x=seasonal["week"],
            y=seasonal["seasonal_high"],
            mode="lines",
            name="80th percentile",
            line=dict(width=0),
            hovertemplate="Week %{x}<br>%{y:.1%}<extra></extra>",
        )
    )
    seasonal_fig.add_trace(
        go.Scatter(
            x=seasonal["week"],
            y=seasonal["seasonal_low"],
            mode="lines",
            name="20th-80th percentile band",
            fill="tonexty",
            line=dict(width=0),
            hovertemplate="Week %{x}<br>%{y:.1%}<extra></extra>",
        )
    )
    seasonal_fig.add_trace(
        go.Scatter(
            x=seasonal["week"],
            y=seasonal["seasonal_avg"],
            mode="lines",
            name="Historical average",
            hovertemplate="Week %{x}<br>%{y:.1%}<extra></extra>",
        )
    )
    seasonal_fig.add_trace(
        go.Scatter(
            x=seasonal["week"],
            y=seasonal["current_year"],
            mode="lines+markers",
            name=f"{latest['date'].year}",
            hovertemplate="Week %{x}<br>%{y:.1%}<extra></extra>",
        )
    )
    seasonal_fig.update_xaxes(title_text="ISO week of year", range=[1, 53], dtick=4)
    seasonal_fig.update_yaxes(title_text="Year-to-date return", tickformat=".0%")
    seasonal_fig.update_layout(
        title=dict(text="Current Year Versus Historical Seasonal Path", x=0, xanchor="left"),
        height=430,
        margin=dict(l=20, r=20, t=55, b=45),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(seasonal_fig, width="stretch")

with tab_forecast:
    interval_fig = go.Figure()
    interval_fig.add_trace(
        go.Bar(
            x=["Next week"],
            y=[forecast.upper - forecast.lower],
            base=[forecast.lower],
            name="80% range",
            hovertemplate="%{base:.2%} to %{y:.2%}<extra></extra>",
        )
    )
    interval_fig.add_trace(
        go.Scatter(
            x=["Next week"],
            y=[forecast.point],
            mode="markers",
            marker=dict(size=14),
            name="Point forecast",
            hovertemplate="%{y:.2%}<extra></extra>",
        )
    )
    interval_fig.update_yaxes(title_text="Next-week return", tickformat=".1%")
    interval_fig.update_layout(
        title=dict(text="Forecast Range", x=0, xanchor="left"),
        height=420,
        margin=dict(l=20, r=20, t=55, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(interval_fig, width="stretch")
    st.write(f"Calibration residual quantile: `{residual_q:.2%}`")

with tab_backtest:
    if visible_bt.empty:
        st.info("No backtest rows available yet.")
    else:
        st.caption(
            f"Backtest rows start after the {BACKTEST_TRAIN_WINDOW}-week rolling "
            "training warm-up; All excludes that warm-up period."
        )
        equity = (1 + visible_bt["strategy_return"]).cumprod()
        equity_fig = go.Figure()
        equity_fig.add_trace(
            go.Scatter(
                x=visible_bt["date"],
                y=equity,
                name="Strategy equity",
                hovertemplate="%{y:.2f}x<extra></extra>",
            )
        )
        style_time_series(equity_fig, "Walk-Forward Strategy Equity", "Growth of $1", 420)
        st.plotly_chart(equity_fig, width="stretch")
        st.dataframe(visible_bt.tail(20), width="stretch")

with tab_memo:
    st.code(generate_crude_memo(latest, forecast, metrics), language="markdown")
