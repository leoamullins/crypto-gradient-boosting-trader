import pandas as pd


def donchian_long_breakout(close: pd.Series, n: int) -> pd.Series:
    """Fixed: Use close prices only to avoid look-ahead bias."""
    upper = close.rolling(n).max()
    return (close > upper.shift(1)).astype(int)


def build_signals(
    prices: pd.DataFrame,
    proba: pd.Series,
    regime: pd.Series,
    breakout_n=20,
    p_enter=0.55,
    p_exit=0.50,
) -> pd.DataFrame:
    """Legacy signal builder for classification mode."""
    sig = prices.copy()
    sig["p"] = proba.reindex(sig.index)
    sig["regime"] = regime.reindex(sig.index, method="ffill").fillna(0)
    sig["bo_long"] = donchian_long_breakout(sig["close"], breakout_n)
    sig["enter"] = (
        (sig["p"] >= p_enter) & (sig["regime"] == 1) & (sig["bo_long"] == 1)
    ).astype(int)
    sig["flat"] = (sig["p"] <= p_exit).astype(int)
    return sig


def build_signals_regression(
    prices: pd.DataFrame,
    predictions: pd.Series,
    regime: pd.Series,
    breakout_n: int = 20,
    pred_threshold: float = 0.0,
    ensemble_scoring: bool = True,
) -> pd.DataFrame:
    """
    Build signals for regression mode with ensemble scoring.

    Args:
        prices: OHLCV DataFrame
        predictions: Predicted returns from regression model
        regime: Regime indicator (0 or 1)
        breakout_n: Donchian breakout window
        pred_threshold: Minimum predicted return to enter
        ensemble_scoring: Use weighted scoring instead of AND logic

    Returns:
        DataFrame with enter/flat signals and position_size
    """
    sig = prices.copy()
    sig["pred"] = predictions.reindex(sig.index).fillna(0)
    sig["regime"] = regime.reindex(sig.index, method="ffill").fillna(0)
    sig["bo_long"] = donchian_long_breakout(sig["close"], breakout_n)

    if ensemble_scoring:
        # Ensemble scoring: combine model + regime + breakout
        # Normalize predicted return to 0-1 range for scoring
        pred_norm = sig["pred"].clip(-0.01, 0.01) / 0.01  # clip to ±1%
        pred_norm = (pred_norm + 1) / 2  # scale to 0-1

        sig["score"] = (
            pred_norm * 2.0  # Model predictions: weight 2.0
            + sig["regime"] * 0.5  # Regime boost: weight 0.5
            + sig["bo_long"] * 0.3  # Breakout boost: weight 0.3
        )

        # Enter when ensemble score is high enough
        sig["enter"] = (sig["score"] > 1.5).astype(int)

        # Position sizing based on predicted return (Kelly-inspired)
        # Size = predicted_return / expected_volatility (simplified)
        sig["position_size"] = sig["pred"].clip(0, 0.01) * 50  # scale to 0-0.5
        sig["position_size"] = sig["position_size"].fillna(0)
    else:
        # Simple threshold-based entry
        sig["enter"] = (
            (sig["pred"] > pred_threshold)
            & (sig["regime"] == 1)
            & (sig["bo_long"] == 1)
        ).astype(int)
        sig["position_size"] = 1.0

    # Exit when model predicts negative returns
    sig["flat"] = (sig["pred"] < 0).astype(int)

    return sig
