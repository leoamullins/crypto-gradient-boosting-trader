# Crypto Gradient Boosting Trader

**In progress**

Gradient-boosted regression signals for BTCUSDT 5‑minute data with ensemble-based trade execution and realistic backtesting.

## Highlights
- Feature-rich dataset combining OHLCV statistics, momentum, microstructure, and macro regime overlays (`src/trader/features`).
- Cost-aware regression targets with walk-forward cross-validation, embargoing, and 20% hold-out testing (`src/trader/models/gbdt.py`).
- Ensemble signal builder that fuses model scores with regime + Donchian breakout context and Kelly-inspired position sizing (`src/trader/strategy/rules.py`).
- Backtester with model-driven exits, optional stop-loss/take-profit, and trade-level analytics (`src/trader/backtester/simulate.py`).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]        # or use poetry based on pyproject.toml
python scripts/backtest_final.py
```
Expected output includes prediction diagnostics, signal counts, and a Sharpe ratio summary (~1.5 in the reference configuration).

## Repository Map
- `scripts/backtest_final.py` – entry point for reproducing the full pipeline end-to-end.
- `src/trader` – modular package covering data loading, feature engineering, labeling, modeling, signal construction, and simulation.
- `data/` – parquet inputs under `data/raw/` plus optional analysis artifacts (e.g., `data/backtest_analysis.png`).
- `tests/` – smoke tests for helpers and indicators.

For a detailed research narrative, see `methodology.md`.
