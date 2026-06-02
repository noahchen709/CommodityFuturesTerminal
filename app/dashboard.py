from __future__ import annotations

from pathlib import Path
import sys

import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import run_walk_forward_backtest, summarize_backtest
from src.data.dataset import load_crude_research_dataset
from src.features.crude_features import add_crude_features, feature_columns
from src.models.conformal import train_conformal_forecaster
from src.reports.memo import generate_crude_memo


st.set_page_config(page_title="Commodity Futures Research Terminal", layout="wide")

st.title("Commodity Futures Research Terminal")
st.caption("Crude oil MVP: supply-demand, positioning, volatility, conformal range, and trade memo.")

raw = load_crude_research_dataset()
features = add_crude_features(raw)
cols = feature_columns()
model, residual_q, forecast = train_conformal_forecaster(features, cols)
bt = run_walk_forward_backtest(features, cols)
metrics = summarize_backtest(bt)
latest = features.iloc[-1]

metric_cols = st.columns(5)
metric_cols[0].metric("WTI settle", f"${latest['settle']:.2f}")
metric_cols[1].metric("Forecast", f"{forecast.point:.2%}")
metric_cols[2].metric("Range", f"{forecast.lower:.2%} to {forecast.upper:.2%}")
metric_cols[3].metric("Vol 8w", f"{latest['vol_8w']:.1%}")
metric_cols[4].metric("Backtest Sharpe", f"{metrics['sharpe']:.2f}")

tab_market, tab_forecast, tab_backtest, tab_memo = st.tabs(
    ["Market", "Forecast Range", "Backtest", "Trading Memo"]
)

with tab_market:
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=features["date"], y=features["settle"], name="WTI settle"))
    price_fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(price_fig, use_container_width=True)

    left, right = st.columns(2)
    left.line_chart(features.set_index("date")["inventory_z_52w"], height=260)
    right.line_chart(features.set_index("date")["positioning_pct_156w"], height=260)

with tab_forecast:
    interval_fig = go.Figure()
    interval_fig.add_trace(
        go.Bar(
            x=["Next week"],
            y=[forecast.upper - forecast.lower],
            base=[forecast.lower],
            name="Conformal interval",
        )
    )
    interval_fig.add_trace(
        go.Scatter(
            x=["Next week"],
            y=[forecast.point],
            mode="markers",
            marker=dict(size=14),
            name="Point forecast",
        )
    )
    interval_fig.update_yaxes(tickformat=".1%")
    interval_fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(interval_fig, use_container_width=True)
    st.write(f"Calibration residual quantile: `{residual_q:.2%}`")

with tab_backtest:
    equity = (1 + bt["strategy_return"]).cumprod()
    equity_fig = go.Figure()
    equity_fig.add_trace(go.Scatter(x=bt["date"], y=equity, name="Strategy equity"))
    equity_fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(equity_fig, use_container_width=True)
    st.dataframe(bt.tail(20), use_container_width=True)

with tab_memo:
    st.code(generate_crude_memo(latest, forecast, metrics), language="markdown")
