import pandas as pd


def compute_regime(df: pd.DataFrame, ma_len: int = 50) -> pd.Series:
    """1 = bull regime, 0 = bear/neutral."""
    ma = df["close"].rolling(ma_len).mean()
    slope = ma.diff()
    regime = ((df["close"] > ma) & (slope > 0)).astype(int)
    regime.name = "regime"
    return regime
