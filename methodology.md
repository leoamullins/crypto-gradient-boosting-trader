# Methodology

This document describes how the trading algorithm is built, trained, and evaluated. It is meant to be a reproducible reference for anyone looking to understand or extend the pipeline implemented across `scripts/backtest_final.py` and the `src/trader` package.

---

## 1. Objective & Scope

- Trade BTCUSDT on 5‑minute bars using a gradient boosted decision tree (XGBoost) regression model that forecasts forward log returns net of transaction costs.
- Use only data available at decision time (no look‑ahead), blend model views with regime and breakout context, and size positions according to expected edge.
- Evaluate performance through a realistic backtest that honors holding periods, transaction costs, and embargoed walk-forward validation.

---

## 2. Data Pipeline

| Component | Description |
| --- | --- |
| Primary market data | `data/raw/BTCUSDT_5m.parquet` containing OHLCV bars sampled every 5 minutes. |
| Regime overlay | `data/raw/BTCUSDT_15m_regime.parquet`, a 15‑minute series that encodes macro trend state (e.g., MA slope). Forward-filled to the 5‑minute grid. |
| Storage helpers | `trader.data.storage.load_parquet` reads Parquet files and enforces a timezone-aware index. |

All downstream features and labels are aligned on the intersection of available timestamps, and rows with missing components introduced by rolling windows are dropped before modeling.

---

## 3. Feature Engineering (`src/trader/features/build.py`)

1. **Returns & Volatility**  
   - Log returns on close, rolling means/stds, and realized ranges across windows 3–48 bars.
   - High/low range compression/expansion and rolling volume z-scores capture regime shifts in liquidity.

2. **Momentum & Oscillators**  
   - RSI (7, 14) and MACD (12/26/9) series to detect oversold/overbought states and cyclical drift.

3. **Microstructure**  
   - Short-horizon imbalance, volatility burst, and volume/return interactions constructed over 3–12 bar windows help the model respond to order-flow shocks.

4. **Calendar Effects**  
   - Hour of day and day-of-week categorical signals allow the model to learn intraday seasonality.

5. **External Regime Alignment**  
   - `attach_regime_features` forward-fills the BTC 15‑minute regime descriptors (state, slope, filters) so each 5‑minute observation carries the latest macro context.

All features are clipped for infinities, cast to float, and left unscaled (tree models handle raw scales). Final feature count is ~70 columns after dropping NaNs.

---

## 4. Label Construction (`src/trader/labels/cost_aware.py`)

1. **Forward Return**  
   - For each bar `t`, compute `fwd_ret = log(close_{t+H} / close_t)` with `H = 12` bars (≈1 hour).

2. **Cost Awareness**  
   - Subtract a realistic round-trip log-return cost of 5 bps (`0.0005`) to obtain the target `y = fwd_ret - cost`.  
   - Training the model on net returns ensures it only recommends trades expected to clear fees.

3. **Regression Mode**  
   - Unlike earlier classification setups, we keep the label continuous so the model can express conviction through magnitude instead of forcing a hard binary decision.

---

## 5. Modeling (`src/trader/models/gbdt.py`)

1. **Model Choice**  
   - `XGBRegressor` with 500 trees, depth 4, learning rate 0.05, 80% subsampling/column sampling, and L2 regularization (`reg_lambda=2`). These defaults favor smooth, generalizable fits on noisy financial data.

2. **Walk-Forward Cross-Validation**  
   - `time_series_splits` produces 20 expanding-window folds that each reserve the most recent ~10% of the seen data as validation while training on prior history.  
   - An embargo of `2 * H` bars removes samples too close to the validation boundary to prevent leakage.

3. **Hold-Out Test Set**  
   - Final 20% of observations are withheld from CV, then scored by a model fit on the entire pre-test history to emulate live trading deployment.

4. **Outputs**  
   - The function returns a timestamp-aligned Series of predicted net returns, available for both training folds and the hold-out segment so downstream logic can use one continuous prediction stream.

5. **Diagnostics**  
   - `scripts/backtest_final.py` logs mean/std of predictions plus MSE and R² on the aligned labels to catch overfitting or degenerate regimes early.

---

## 6. Signal Construction (`src/trader/strategy/rules.py`)

1. **Inputs**  
   - `prices`: original OHLCV slice aligned to predictions.  
   - `predictions`: net-return forecasts.  
   - `regime`: 0/1 macro trend state.  
   - `donchian_long_breakout`: 20-bar close-only breakout indicator.

2. **Ensemble Scoring**  
   - Predictions are clipped to ±1% and normalized to 0–1, then combined as  
     `score = 2.0 * pred_norm + 0.5 * regime + 0.3 * breakout`.  
   - This replaces brittle AND logic with a weighted additive view that keeps promising trades even if one filter is neutral.

3. **Entries & Position Sizing**  
   - Enter when `score > 1.5`.  
   - Position size is proportional to predicted edge: `size = clip(pred, 0, 1%) * 50`, capped by `max_position=1.0`. A 1% predicted net return therefore maps to a half-size trade (0.5 notional), limiting leverage until conviction is high.

4. **Exits**  
   - `flat = (pred < 0)` triggers an exit signal whenever the model expects negative returns, allowing dynamic, data-driven trade termination.

---

## 7. Execution & Backtesting (`src/trader/backtester/simulate.py`)

1. **Trade Loop**  
   - Iterate through signals, open a position when `enter=1` and no active trade.  
   - Track entry price/time, position size, and evaluate exit criteria each bar.

2. **Exit Hierarchy**  
   - Stops disabled in final configuration (`use_stops=False`) because model-guided exits proved superior.  
   - Otherwise, stop-loss at `-2 × cost` and take-profit at `+4 × cost`.  
   - Always enforce model-based exit (`flat=1`) and a hard time stop at `H` bars.

3. **P&L Calculation**  
   - Log-return between entry and exit minus cost, scaled by position size.  
   - Stores trade-by-trade returns, position sizes, and the exit reason for later diagnostics.

4. **Performance Metrics**  
   - Sharpe ratio computed over the distribution of trades, annualized using observed trades/year to avoid earlier scaling bugs.  
   - Script reports win rate, win/loss ratio, and descriptive stats. Resulting configuration (regression + ensemble + sizing) yields ~2k trades and Sharpe ≈ 1.53 on historical BTC data.

---

## 8. Reproduction Checklist

1. Ensure parquet files under `data/raw/` are present (see Section 2).  
2. Install dependencies: `pip install -e .[dev]` or use `poetry` via `pyproject.toml`.  
3. Run the reference experiment:
   ```bash
   python scripts/backtest_final.py
   ```
4. Review console output and `data/backtest_analysis.png` for equity and distribution plots (generated externally).  
5. Iterate on configs (holding period `H`, cost assumptions, scoring weights) inside `scripts/backtest_final.py` to test hypotheses.

---

## 9. Assumptions & Limitations

- The model only sees BTC; generalization to other assets requires retraining with asset-specific cost and regime inputs.
- Transaction costs are static; slippage, funding, and borrow fees are not modeled.
- Position sizing is proportional to predicted edge but does not cap aggregate portfolio leverage; integrate with account-level risk controls before live trading.
- Walk-forward validation guards against simple leakage but does not simulate liquidity or order-book constraints—results should be viewed as indicative, not executable PnL.

---

This methodology will stay aligned with the codebase; update it whenever data sources, features, or execution logic change to keep the research record trustworthy and reproducible.
