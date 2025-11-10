"""
Generate comprehensive PnL visualizations for the improved backtest.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from trader.data.storage import load_parquet
from trader.features.build import make_features_5m_15m
from trader.labels.cost_aware import add_cost_aware_label
from trader.models.gbdt import walkforward_predict
from trader.strategy.rules import build_signals_regression
from trader.backtester.simulate import run_backtest

# Set style
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["font.size"] = 10

# ===== RUN BACKTEST =====
print("Running backtest...")

SYMBOL_FILE = "raw/BTCUSDT_5m.parquet"
BTC_REGIME_FILE = "raw/BTCUSDT_15m_regime.parquet"
BREAKOUT_N = 20
H = 12
COST = 0.0005

df = load_parquet(SYMBOL_FILE)
regime = load_parquet(BTC_REGIME_FILE)["regime"]

feat = make_features_5m_15m(df, regime)
feat = feat.join(df[["close"]]).dropna()
feat = add_cost_aware_label(
    feat, H=H, roundtrip_cost_logret=COST, mode="regression"
).dropna()

X = feat.drop(columns=["y", "fwd_ret"])
y = feat["y"]

predictions = walkforward_predict(
    X, y, embargo=H * 2, n_splits=20, mode="regression", test_size=0.2
).dropna()

aligned = df.loc[predictions.index]
sigs = build_signals_regression(
    aligned, predictions, regime, breakout_n=BREAKOUT_N, ensemble_scoring=True
)

stats = run_backtest(
    sigs,
    H=H,
    roundtrip_cost_logret=COST,
    use_stops=False,
    use_position_sizing=True,
    max_position=1.0,
)

print(f"Trades: {len(stats)}")
print(f"Sharpe: {float(stats['sharpe_over_trades'].iloc[0]):.4f}")

# ===== CREATE VISUALIZATIONS =====
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# 1. Cumulative PnL
ax1 = fig.add_subplot(gs[0, :])
cumulative_pnl = stats["pnl"].cumsum()
ax1.plot(cumulative_pnl.values, linewidth=2, color="#2E86AB")
ax1.fill_between(
    range(len(cumulative_pnl)), 0, cumulative_pnl.values, alpha=0.3, color="#2E86AB"
)
ax1.axhline(y=0, color="red", linestyle="--", alpha=0.5)
ax1.set_title("Cumulative P&L Over Time", fontsize=14, fontweight="bold")
ax1.set_xlabel("Trade Number")
ax1.set_ylabel("Cumulative P&L")
ax1.grid(True, alpha=0.3)

# Add annotations
final_pnl = cumulative_pnl.iloc[-1]
ax1.text(
    0.02,
    0.95,
    f"Final P&L: {final_pnl:.6f}",
    transform=ax1.transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

# 2. PnL Distribution
ax2 = fig.add_subplot(gs[1, 0])
pnl = stats["pnl"]
winners = pnl[pnl > 0]
losers = pnl[pnl <= 0]

ax2.hist(
    winners,
    bins=50,
    alpha=0.7,
    color="green",
    label=f"Winners ({len(winners)})",
    edgecolor="black",
)
ax2.hist(
    losers,
    bins=50,
    alpha=0.7,
    color="red",
    label=f"Losers ({len(losers)})",
    edgecolor="black",
)
ax2.axvline(
    x=pnl.mean(),
    color="blue",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {pnl.mean():.6f}",
)
ax2.axvline(x=0, color="black", linestyle="-", linewidth=1)
ax2.set_title("P&L Distribution", fontsize=12, fontweight="bold")
ax2.set_xlabel("P&L")
ax2.set_ylabel("Frequency")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Box Plot: Winners vs Losers
ax3 = fig.add_subplot(gs[1, 1])
data_to_plot = [winners.values, losers.values]
bp = ax3.boxplot(
    data_to_plot,
    labels=["Winners", "Losers"],
    patch_artist=True,
    showmeans=True,
    meanline=True,
)
bp["boxes"][0].set_facecolor("lightgreen")
bp["boxes"][1].set_facecolor("lightcoral")
ax3.set_title("Win/Loss Box Plot", fontsize=12, fontweight="bold")
ax3.set_ylabel("P&L")
ax3.grid(True, alpha=0.3, axis="y")

# Add stats text
win_rate = len(winners) / len(pnl) * 100
wl_ratio = abs(winners.mean() / losers.mean()) if len(losers) > 0 else 0
ax3.text(
    0.5,
    0.02,
    f"Win Rate: {win_rate:.1f}%\nW/L Ratio: {wl_ratio:.2f}",
    transform=ax3.transAxes,
    fontsize=10,
    verticalalignment="bottom",
    horizontalalignment="center",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

# 4. Exit Reasons Pie Chart
ax4 = fig.add_subplot(gs[1, 2])
exit_counts = stats["exit_reason"].value_counts()
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
wedges, texts, autotexts = ax4.pie(
    exit_counts.values,
    labels=exit_counts.index,
    autopct="%1.1f%%",
    colors=colors,
    startangle=90,
)
for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_fontweight("bold")
ax4.set_title("Exit Reasons", fontsize=12, fontweight="bold")

# 5. Rolling Sharpe (50-trade window)
ax5 = fig.add_subplot(gs[2, :2])
window = 50
if len(pnl) > window:
    rolling_mean = pnl.rolling(window).mean()
    rolling_std = pnl.rolling(window).std()
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(window)

    ax5.plot(rolling_sharpe.values, linewidth=2, color="#6A4C93")
    ax5.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax5.axhline(y=1, color="green", linestyle="--", alpha=0.5, label="Sharpe = 1.0")
    ax5.fill_between(
        range(len(rolling_sharpe)),
        0,
        rolling_sharpe.values,
        where=(rolling_sharpe.values > 0),
        alpha=0.3,
        color="green",
    )
    ax5.fill_between(
        range(len(rolling_sharpe)),
        0,
        rolling_sharpe.values,
        where=(rolling_sharpe.values <= 0),
        alpha=0.3,
        color="red",
    )
    ax5.set_title(
        f"Rolling Sharpe Ratio ({window}-trade window)", fontsize=12, fontweight="bold"
    )
    ax5.set_xlabel("Trade Number")
    ax5.set_ylabel("Rolling Sharpe")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

# 6. Position Size Distribution
ax6 = fig.add_subplot(gs[2, 2])
if "position_size" in stats.columns:
    position_sizes = stats["position_size"]
    ax6.hist(position_sizes, bins=30, color="#F77F00", edgecolor="black", alpha=0.7)
    ax6.axvline(
        x=position_sizes.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {position_sizes.mean():.3f}",
    )
    ax6.set_title("Position Size Distribution", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Position Size")
    ax6.set_ylabel("Frequency")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

# 7. Drawdown Analysis
ax7 = fig.add_subplot(gs[3, :2])
cumulative = cumulative_pnl.values
running_max = np.maximum.accumulate(cumulative)
drawdown = cumulative - running_max

ax7.fill_between(range(len(drawdown)), 0, drawdown, color="red", alpha=0.5)
ax7.plot(drawdown, color="darkred", linewidth=1.5)
ax7.set_title("Drawdown", fontsize=12, fontweight="bold")
ax7.set_xlabel("Trade Number")
ax7.set_ylabel("Drawdown")
ax7.grid(True, alpha=0.3)

max_dd = drawdown.min()
max_dd_idx = drawdown.argmin()
ax7.annotate(
    f"Max DD: {max_dd:.6f}",
    xy=(max_dd_idx, max_dd),
    xytext=(max_dd_idx + 100, max_dd + 0.005),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
)

# 8. Summary Statistics Table
ax8 = fig.add_subplot(gs[3, 2])
ax8.axis("off")

sharpe = float(stats["sharpe_over_trades"].iloc[0])
total_trades = len(pnl)
total_pnl = pnl.sum()
mean_pnl = pnl.mean()
std_pnl = pnl.std()
max_win = pnl.max()
max_loss = pnl.min()
win_rate = len(winners) / total_trades * 100
avg_win = winners.mean() if len(winners) > 0 else 0
avg_loss = losers.mean() if len(losers) > 0 else 0

summary_text = f"""
SUMMARY STATISTICS
{'='*35}
Total Trades:        {total_trades:,}
Sharpe Ratio:        {sharpe:.4f}
Total P&L:           {total_pnl:.6f}
Mean P&L:            {mean_pnl:.6f}
Std Dev:             {std_pnl:.6f}

Win Rate:            {win_rate:.1f}%
Avg Winner:          {avg_win:.6f}
Avg Loser:           {avg_loss:.6f}
W/L Ratio:           {wl_ratio:.4f}

Best Trade:          {max_win:.6f}
Worst Trade:         {max_loss:.6f}
Max Drawdown:        {max_dd:.6f}
"""

ax8.text(
    0.1,
    0.9,
    summary_text,
    fontsize=11,
    verticalalignment="top",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
)

# Overall title
fig.suptitle(
    "Crypto Gradient Boosting Trader - Performance Analysis",
    fontsize=18,
    fontweight="bold",
    y=0.995,
)

# Save figure
output_file = "data/backtest_analysis.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"\n✅ Visualization saved to: {output_file}")

plt.show()
