import pandas as pd


def donchian_long_breakout(close: pd.Series, n: int) -> pd.Series:
    """Donchian channel breakout signal."""
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
    """Generate entry/exit signals and position sizes from regression predictions."""
    sig = prices.copy()
    sig["pred"] = predictions.reindex(sig.index).fillna(0)
    sig["regime"] = regime.reindex(sig.index, method="ffill").fillna(0)
    sig["bo_long"] = donchian_long_breakout(sig["close"], breakout_n)

    if ensemble_scoring:
        pred_norm = sig["pred"].clip(-0.01, 0.01) / 0.01
        pred_norm = (pred_norm + 1) / 2

        sig["score"] = (
            pred_norm * 2.0
            + sig["regime"] * 0.5
            + sig["bo_long"] * 0.3
        )

        sig["enter"] = (sig["score"] > 1.5).astype(int)
        sig["position_size"] = sig["pred"].clip(0, 0.01) * 50
        sig["position_size"] = sig["position_size"].fillna(0)
    else:
        sig["enter"] = (
            (sig["pred"] > pred_threshold)
            & (sig["regime"] == 1)
            & (sig["bo_long"] == 1)
        ).astype(int)
        sig["position_size"] = 1.0

    sig["flat"] = (sig["pred"] < 0).astype(int)

    return sig
