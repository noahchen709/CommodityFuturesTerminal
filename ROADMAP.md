# Project Roadmap

This project is a commodity futures research terminal first, and a trading strategy second. The goal is to build a credible, transparent workflow for crude oil research: clean data, explainable features, honest validation, uncertainty-aware forecasts, and risk-managed trade signals.

The project should be presented as a live research lab rather than a claim that the model can predict oil prices. Profitability is a target to investigate, but the core value is showing disciplined research judgment.

## Guiding Principles

- Treat every model as a hypothesis that must beat simple baselines.
- Use walk-forward validation for all market results.
- Report uncertainty, calibration, drawdowns, and transaction costs.
- Prefer interpretable models until the data supports more complexity.
- Keep a research log so each improvement can become a LinkedIn update.

## Near-Term Technical Tasks

1. Add a model leaderboard.
   Compare zero-return, historical mean, Ridge, tuned Ridge, ElasticNet, logistic regression, calibrated logistic regression, random forest, gradient boosting, and simple rule-based strategies.

2. Add richer model metrics.
   Track MAE, RMSE, directional accuracy, AUC, Brier score, calibration error, Sharpe, Sortino, max drawdown, turnover, hit rate, average win/loss, and cost-adjusted returns.

3. Add transaction costs and slippage.
   Backtests should show both raw and cost-adjusted performance. This is the first real test of whether a signal might be tradable.

4. Tune the current models.
   The current Ridge and logistic regression settings are reasonable baselines, but they should be selected through walk-forward validation rather than hard-coded.

5. Add probability calibration.
   Predicted probabilities should be checked against realized frequencies. If the model says 70% probability of an up week, the historical realized up rate should be close to 70%.

6. Improve seasonal encoding.
   Replace numeric month and quarter features with cyclic encodings or one-hot features so the model does not treat calendar periods as linear quantities.

## Profitability Research Tasks

1. Add realistic trading frictions.
   Include commission assumptions, bid/ask slippage, turnover costs, and futures roll considerations.

2. Add position sizing.
   Test fixed size, volatility targeting, confidence-weighted sizing, and max exposure limits.

3. Add risk controls.
   Include max weekly loss guardrails, drawdown limits, stop-loss logic, take-profit logic, and flat regimes when uncertainty is high.

4. Compare strategy modes.
   Test long-only, short-only, long/short, and flat-allowed strategies separately.

5. Add robustness tests.
   Evaluate performance across different periods, including inflationary markets, oil shocks, low-volatility regimes, high-volatility regimes, and recent out-of-sample data.

## Feature Engineering Tasks

1. Add futures curve features.
   Include contango/backwardation, 1-month spreads, 3-month spreads, 6-month spreads, 12-month spreads, and roll yield.

2. Improve inventory features.
   Add inventory level, inventory change, inventory surprise versus seasonal average, and storage regime indicators.

3. Add supply and demand indicators.
   Include production, refinery utilization, imports, exports, product demand, and crack spreads where reliable data is available.

4. Add cross-market features.
   Include Brent-WTI spread, dollar trend, rates, real yields, volatility indexes, and broad risk sentiment.

5. Add event flags.
   Track OPEC meetings, major inventory shocks, geopolitical stress periods, and extreme volatility weeks.

## Dashboard Tasks

1. Add a model leaderboard view.
   Show performance metrics for each model and make the current production choice explicit.

2. Add model diagnostics.
   Include calibration plots, feature importance, residual distributions, error by regime, and forecast-vs-realized charts.

3. Add a trade log.
   Show each signal, realized return, strategy return, cost-adjusted return, and reason for entering or staying flat.

4. Add risk views.
   Include equity curve, drawdown curve, exposure over time, turnover, active trade rate, and rolling Sharpe.

5. Add source health checks.
   Show data freshness, missing source warnings, and fallback/cached data status.

## LinkedIn Update Plan

1. Announce the project.
   Explain that this is a commodity futures research terminal focused on uncertainty-aware crude oil signals.

2. Share the first model audit.
   Show that the initial Ridge point forecast was weak and explain why honest baselines matter.

3. Add walk-forward validation.
   Explain why random train/test splits are inappropriate for financial time series.

4. Add transaction costs.
   Show how raw strategy performance changes after costs and slippage.

5. Add model benchmarking.
   Compare simple baselines, linear models, and tree-based models.

6. Add term-structure features.
   Explain contango, backwardation, and why the futures curve matters in commodities.

7. Add probability calibration.
   Show whether model probabilities match realized outcomes.

8. Publish a research review.
   Summarize what improved, what failed, whether the signal looks tradable, and what the next experiment will test.

## Suggested Priority Order

1. Add transaction costs and slippage.
2. Add a model benchmark and leaderboard.
3. Tune Ridge and logistic regression with walk-forward validation.
4. Add probability calibration.
5. Add futures curve features.
6. Add model diagnostics to the dashboard.
7. Add risk-managed strategy logic.
8. Improve the README with screenshots, methodology, and limitations.
9. Start posting weekly LinkedIn research updates.
10. Test more complex models only after the baseline framework is trustworthy.
