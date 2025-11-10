import numpy as np
import pandas as pd


def run_backtest(
    signals: pd.DataFrame,
    H: int,
    roundtrip_cost_logret: float,
    stop_loss_mult: float = 2.0,
    take_profit_mult: float = 4.0,
    use_stops: bool = True,
    use_position_sizing: bool = True,
    max_position: float = 1.0,
    min_hold_bars: int = 3,
) -> pd.DataFrame:
    """Run backtest simulation with position sizing and stop logic."""
    c = signals["close"]
    pnl, pos, entry_price, entry_idx, position_sizes = [], 0, None, None, []
    exit_reasons = []
    has_position_size = "position_size" in signals.columns

    for i in range(len(signals) - H - 1):
        if pos == 0 and signals["enter"].iat[i] == 1:
            if use_position_sizing and has_position_size:
                pos_size = min(abs(signals["position_size"].iat[i]), max_position)
            else:
                pos_size = max_position
            pos, entry_price, entry_idx = pos_size, c.iat[i], i
            continue

        exit_flag = False
        exit_reason = None

        if entry_idx is not None and (i - entry_idx < min_hold_bars):
            continue  # skip exit checks

        if pos > 0:
            current_price = c.iat[i]
            current_return = np.log(current_price / entry_price)

            if use_stops:
                if current_return < -stop_loss_mult * roundtrip_cost_logret:
                    exit_flag = True
                    exit_reason = "stop_loss"
                elif current_return > take_profit_mult * roundtrip_cost_logret:
                    exit_flag = True
                    exit_reason = "take_profit"

            if signals["flat"].iat[i] == 1 and not exit_flag:
                exit_flag = True
                exit_reason = "model_exit"

            if entry_idx is not None and (i - entry_idx >= H) and not exit_flag:
                exit_flag = True
                exit_reason = "time_limit"

        if pos > 0 and exit_flag:
            r = (np.log(c.iat[i] / entry_price) - roundtrip_cost_logret) * pos
            pnl.append(r)
            position_sizes.append(pos)
            exit_reasons.append(exit_reason)
            pos, entry_price, entry_idx = 0, None, None

    pnl = pd.Series(pnl, name="trade_ret")

    if len(pnl) > 5:
        years = len(signals) / (365.25 * 288)
        trades_per_year = len(pnl) / years if years > 0 else 252
        sharpe = pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(trades_per_year)
    else:
        sharpe = np.nan

    return pd.DataFrame(
        {"pnl": pnl, "position_size": position_sizes, "exit_reason": exit_reasons}
    ).assign(sharpe_over_trades=sharpe)
