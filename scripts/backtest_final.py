"""
FINAL OPTIMIZED BACKTEST - REGRESSION MODE
===========================================
Uses all high-priority improvements:
1. Regression target (predicts net returns, not binary)
2. 20-fold walk-forward CV with 20% hold-out test set
3. Model-based exits (NO fixed stop-loss/take-profit)
4. Position sizing based on predicted returns
5. Ensemble scoring (model + regime + breakout)
"""

import pandas as pd
from trader.data.storage import load_parquet
from trader.features.build import make_features_5m_15m
from trader.labels.cost_aware import add_cost_aware_label
from trader.models.gbdt import walkforward_predict
from trader.strategy.rules import build_signals_regression
from trader.backtester.simulate import run_backtest

# ===== CONFIGURATION =====
SYMBOL_FILE = "raw/BTCUSDT_5m.parquet"
BTC_REGIME_FILE = "raw/BTCUSDT_15m_regime.parquet"
BREAKOUT_N = 20
H = 12  # 1 hour holding period
COST = 0.0005  # 5 bps - realistic exchange fees

# Model-based exits only (NO fixed stops)
USE_STOPS = False
USE_POSITION_SIZING = True
MAX_POSITION = 1.0

print("=" * 70)
print("FINAL OPTIMIZED BACKTEST")
print("=" * 70)

# ===== LOAD DATA =====
print("\n[1/5] Loading data...")
df = load_parquet(SYMBOL_FILE)
regime = load_parquet(BTC_REGIME_FILE)["regime"]
print(f"  Loaded {len(df):,} bars of BTCUSDT 5-minute data")

# ===== FEATURE ENGINEERING =====
print("\n[2/5] Building features...")
feat = make_features_5m_15m(df, regime)
feat = feat.join(df[["close"]]).dropna()
feat = add_cost_aware_label(feat, H=H, roundtrip_cost_logret=COST, mode="regression").dropna()
print(f"  Total samples: {len(feat):,}")
print(f"  Features: {len([c for c in feat.columns if c not in ['y', 'fwd_ret', 'close']])}")

# ===== MODEL TRAINING =====
print("\n[3/5] Training regression model...")
print("  - 20-fold walk-forward cross-validation")
print("  - 20% hold-out test set")
print("  - Embargo: 2x holding period")

X = feat.drop(columns=["y", "fwd_ret"])
y = feat["y"]

predictions = walkforward_predict(
    X, y, embargo=H * 2, n_splits=20, mode="regression", test_size=0.2
).dropna()

print(f"  ✓ Predictions: {len(predictions):,}")
print(f"    Mean: {predictions.mean():.6f}")
print(f"    Std: {predictions.std():.6f}")

# Check model performance
from sklearn.metrics import mean_squared_error, r2_score
y_aligned = y.loc[predictions.index]
mse = mean_squared_error(y_aligned, predictions)
r2 = r2_score(y_aligned, predictions)
print(f"  Model Performance:")
print(f"    MSE: {mse:.8f}")
print(f"    R²: {r2:.4f}")

# ===== SIGNAL GENERATION =====
print("\n[4/5] Generating signals...")
print("  - Ensemble scoring (model + regime + breakout)")
print("  - Position sizing based on predicted returns")

aligned = df.loc[predictions.index]
sigs = build_signals_regression(
    aligned, predictions, regime, breakout_n=BREAKOUT_N, ensemble_scoring=True
)

enter_count = sigs['enter'].sum()
print(f"  ✓ Enter signals: {enter_count:,}")
if USE_POSITION_SIZING and enter_count > 0:
    avg_size = sigs.loc[sigs['enter']==1, 'position_size'].mean()
    print(f"    Avg position size: {avg_size:.3f}")

# ===== BACKTEST =====
print("\n[5/5] Running backtest...")
print(f"  - Model-based exits only (no fixed stops)")
print(f"  - Max holding period: {H} bars (1 hour)")
print(f"  - Transaction cost: {COST*10000:.1f} bps")

stats = run_backtest(
    sigs,
    H=H,
    roundtrip_cost_logret=COST,
    use_stops=USE_STOPS,
    use_position_sizing=USE_POSITION_SIZING,
    max_position=MAX_POSITION,
)

# ===== RESULTS =====
print("\n" + "=" * 70)
print("BACKTEST RESULTS")
print("=" * 70)

n_trades = len(stats["pnl"])
print(f"\n📊 Total Trades: {n_trades:,}")

if n_trades > 5:
    sharpe = float(stats["sharpe_over_trades"].iloc[0])
    pnl = stats["pnl"]

    # Summary metrics
    print(f"\n💰 Performance Metrics:")
    print(f"  Sharpe Ratio: {sharpe:.4f}")
    print(f"  Total P&L: {pnl.sum():.6f}")
    print(f"  Mean P&L per trade: {pnl.mean():.6f}")
    print(f"  Median P&L: {pnl.median():.6f}")
    print(f"  Std Dev: {pnl.std():.6f}")

    # Win/Loss Analysis
    winners = pnl[pnl > 0]
    losers = pnl[pnl <= 0]

    print(f"\n📈 Win/Loss Analysis:")
    print(f"  Winners: {len(winners):,} ({100*len(winners)/n_trades:.1f}%)")
    print(f"    Avg: {winners.mean():.6f}")
    print(f"    Total: {winners.sum():.6f}")
    print(f"  Losers: {len(losers):,} ({100*len(losers)/n_trades:.1f}%)")
    print(f"    Avg: {losers.mean():.6f}")
    print(f"    Total: {losers.sum():.6f}")

    if len(winners) > 0 and len(losers) > 0:
        wl_ratio = abs(winners.mean() / losers.mean())
        print(f"  Avg Win/Loss Ratio: {wl_ratio:.4f}")

    # Exit reasons
    print(f"\n🚪 Exit Reasons:")
    exit_counts = stats["exit_reason"].value_counts()
    for reason, count in exit_counts.items():
        print(f"  {reason:15s}: {count:5,} ({100*count/n_trades:5.1f}%)")

    # Distribution
    print(f"\n📉 P&L Distribution:")
    print(pnl.describe().to_string())

    # Performance verdict
    print(f"\n" + "=" * 70)
    if sharpe > 1.0:
        print("✅ PROFITABLE STRATEGY (Sharpe > 1.0)")
    elif sharpe > 0:
        print("⚠️  MARGINALLY PROFITABLE (0 < Sharpe < 1.0)")
    else:
        print("❌ UNPROFITABLE STRATEGY (Sharpe < 0)")
    print("=" * 70)

else:
    print("❌ Not enough trades to evaluate!")
