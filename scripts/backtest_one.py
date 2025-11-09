import pandas as pd
from trader.data.storage import load_parquet
from trader.features.build import make_features_5m_15m
from trader.labels.cost_aware import add_cost_aware_label
from trader.models.gbdt import walkforward_proba
from trader.strategy.rules import build_signals
from trader.backtester.simulate import run_backtest

# inputs
SYMBOL_FILE = "raw/BTCUSDT_5m.parquet"
BTC_REGIME_FILE = "raw/BTCUSDT_15m_regime.parquet"
BREAKOUT_N = 20
H = 12  # Increased from 3 to 12 bars (1 hour vs 15 min)
COST = 0.0005  # Changed from 0.0025 (25bps) to 0.0005 (5bps) - realistic exchange fees
P_ENTER = 0.20  # Lowered from 0.55 to match model calibration
P_EXIT = 0.10  # Must be < P_ENTER to avoid immediate exits

df = load_parquet(SYMBOL_FILE)
regime = load_parquet(BTC_REGIME_FILE)["regime"]

# features + labels
feat = make_features_5m_15m(df, regime)
# keep price column for labeling / strategy alignment
feat = feat.join(df[["close"]]).dropna()
feat = add_cost_aware_label(feat, H=H, roundtrip_cost_logret=COST).dropna()

X = feat.drop(columns=["y", "fwd_ret"])
y = feat["y"]

# CV proba
proba = walkforward_proba(X, y, embargo=max(48, BREAKOUT_N)).dropna()

# signals + backtest
aligned = df.loc[proba.index]  # prices aligned to proba index
sigs = build_signals(
    aligned, proba, regime, breakout_n=BREAKOUT_N, p_enter=P_ENTER, p_exit=P_EXIT
)
stats = run_backtest(sigs, H=H, roundtrip_cost_logret=COST)

print("Trades:", len(stats["pnl"]))
print("Sharpe_over_trades:", float(stats["sharpe_over_trades"].iloc[0]))
print(stats["pnl"].describe())
