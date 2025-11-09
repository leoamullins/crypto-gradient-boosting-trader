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
MODE = "regression"  # Use regression instead of classification

# Stop-loss/take-profit parameters
STOP_LOSS_MULT = 2.0  # Exit at -2x cost
TAKE_PROFIT_MULT = 4.0  # Exit at +4x cost

# Position sizing
USE_POSITION_SIZING = True
MAX_POSITION = 1.0

print("=" * 70)
print("IMPROVED BACKTEST - REGRESSION MODE WITH STOP-LOSS/TAKE-PROFIT")
print("=" * 70)

# ===== LOAD DATA =====
print("\nLoading data...")
df = load_parquet(SYMBOL_FILE)
regime = load_parquet(BTC_REGIME_FILE)["regime"]

# ===== FEATURE ENGINEERING =====
print("Building features...")
feat = make_features_5m_15m(df, regime)
feat = feat.join(df[["close"]]).dropna()
feat = add_cost_aware_label(feat, H=H, roundtrip_cost_logret=COST, mode=MODE).dropna()

print(f"  Total samples: {len(feat):,}")
print(f"  Features: {len([c for c in feat.columns if c not in ['y', 'fwd_ret', 'close']])}")

# ===== MODEL TRAINING =====
print("\nTraining regression model with walk-forward CV...")
X = feat.drop(columns=["y", "fwd_ret"])
y = feat["y"]

print(f"  Label stats: mean={y.mean():.6f}, std={y.std():.6f}")

predictions = walkforward_predict(
    X, y, embargo=H * 2, n_splits=20, mode=MODE, test_size=0.2
).dropna()

print(f"  Predictions generated: {len(predictions):,}")
print(f"  Prediction stats: mean={predictions.mean():.6f}, std={predictions.std():.6f}")

# ===== SIGNAL GENERATION =====
print("\nGenerating signals with ensemble scoring...")
aligned = df.loc[predictions.index]
sigs = build_signals_regression(
    aligned, predictions, regime, breakout_n=BREAKOUT_N, ensemble_scoring=True
)

print(f"  Enter signals: {sigs['enter'].sum():,}")
if USE_POSITION_SIZING:
    print(f"  Avg position size: {sigs.loc[sigs['enter']==1, 'position_size'].mean():.3f}")

# ===== BACKTEST =====
print("\nRunning backtest with stop-loss/take-profit...")
stats = run_backtest(
    sigs,
    H=H,
    roundtrip_cost_logret=COST,
    stop_loss_mult=STOP_LOSS_MULT,
    take_profit_mult=TAKE_PROFIT_MULT,
    use_stops=True,
    use_position_sizing=USE_POSITION_SIZING,
    max_position=MAX_POSITION,
)

# ===== RESULTS =====
print("\n" + "=" * 70)
print("BACKTEST RESULTS")
print("=" * 70)

n_trades = len(stats["pnl"])
print(f"\nTotal Trades: {n_trades}")

if n_trades > 0:
    sharpe = float(stats["sharpe_over_trades"].iloc[0])
    print(f"Sharpe Ratio: {sharpe:.4f}")

    pnl = stats["pnl"]
    winners = pnl[pnl > 0]
    losers = pnl[pnl <= 0]

    print(f"\nP&L Statistics:")
    print(f"  Mean: {pnl.mean():.6f}")
    print(f"  Median: {pnl.median():.6f}")
    print(f"  Std: {pnl.std():.6f}")
    print(f"  Total: {pnl.sum():.6f}")

    print(f"\nWin/Loss Analysis:")
    print(f"  Winners: {len(winners)} ({100*len(winners)/n_trades:.1f}%)")
    print(f"    Avg: {winners.mean():.6f}")
    print(f"    Total: {winners.sum():.6f}")
    print(f"  Losers: {len(losers)} ({100*len(losers)/n_trades:.1f}%)")
    print(f"    Avg: {losers.mean():.6f}")
    print(f"    Total: {losers.sum():.6f}")

    if len(winners) > 0 and len(losers) > 0:
        print(f"\nWin/Loss Ratio: {abs(winners.mean() / losers.mean()):.4f}")

    # Exit reasons
    print(f"\nExit Reasons:")
    exit_counts = stats["exit_reason"].value_counts()
    for reason, count in exit_counts.items():
        print(f"  {reason}: {count} ({100*count/n_trades:.1f}%)")

    print(f"\nDetailed PnL Distribution:")
    print(pnl.describe())

else:
    print("No trades executed!")

print("\n" + "=" * 70)
