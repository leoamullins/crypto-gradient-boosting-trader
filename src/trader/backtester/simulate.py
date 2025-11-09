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
) -> pd.DataFrame:
    """
    Run backtest with improved exit logic and position sizing.

    Args:
        signals: DataFrame with enter/flat signals, close prices, and optional 'position_size' column
        H: Maximum holding period in bars
        roundtrip_cost_logret: Transaction cost in log-return units
        stop_loss_mult: Stop loss as multiple of cost (e.g., 2.0 = exit at -2x cost)
        take_profit_mult: Take profit as multiple of cost (e.g., 4.0 = exit at +4x cost)
        use_stops: Whether to use stop-loss/take-profit (if False, uses old logic)
        use_position_sizing: Whether to use variable position sizing
        max_position: Maximum position size (1.0 = full size)
    """
    c = signals["close"]
    pnl, pos, entry_price, entry_idx, position_sizes = [], 0, None, None, []
    exit_reasons = []

    # Check if position_size column exists
    has_position_size = "position_size" in signals.columns

    for i in range(len(signals) - H - 1):
        if pos == 0 and signals["enter"].iat[i] == 1:
            # Determine position size
            if use_position_sizing and has_position_size:
                pos_size = min(abs(signals["position_size"].iat[i]), max_position)
            else:
                pos_size = max_position

            pos, entry_price, entry_idx = pos_size, c.iat[i], i

        exit_flag = False
        exit_reason = None

        if pos > 0:
            current_price = c.iat[i]
            current_return = np.log(current_price / entry_price)

            if use_stops:
                # Stop-loss: exit if losing more than stop_loss_mult * cost
                if current_return < -stop_loss_mult * roundtrip_cost_logret:
                    exit_flag = True
                    exit_reason = "stop_loss"

                # Take-profit: exit if gaining more than take_profit_mult * cost
                elif current_return > take_profit_mult * roundtrip_cost_logret:
                    exit_flag = True
                    exit_reason = "take_profit"

            # Model-based exit (flat signal)
            if signals["flat"].iat[i] == 1 and not exit_flag:
                exit_flag = True
                exit_reason = "model_exit"

            # Time-based exit (max holding period)
            if entry_idx is not None and (i - entry_idx >= H) and not exit_flag:
                exit_flag = True
                exit_reason = "time_limit"

        if pos > 0 and exit_flag:
            # P&L scaled by position size
            r = (np.log(c.iat[i] / entry_price) - roundtrip_cost_logret) * pos
            pnl.append(r)
            position_sizes.append(pos)
            exit_reasons.append(exit_reason)
            pos, entry_price, entry_idx = 0, None, None

    pnl = pd.Series(pnl, name="trade_ret")

    # Fixed: Use proper annualization for trade returns
    if len(pnl) > 5:
        years = len(signals) / (365.25 * 288)  # 5-min bars
        trades_per_year = len(pnl) / years if years > 0 else 252
        sharpe = pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(trades_per_year)
    else:
        sharpe = np.nan

    return pd.DataFrame(
        {"pnl": pnl, "position_size": position_sizes, "exit_reason": exit_reasons}
    ).assign(sharpe_over_trades=sharpe)
