from __future__ import annotations
import pandas as pd
import numpy as np
from .ta import logret, rsi, macd, rolling_stats, hl_range, vol_features, microstructure


def add_calendar(df: pd.DataFrame, tz: str | None = None) -> pd.DataFrame:
    idx = df.index if tz is None else df.index.tz_convert(tz)
    return pd.DataFrame(
        {
            "hour": idx.hour,
            "dow": idx.dayofweek,
        },
        index=df.index,
    )


def make_features_core(
    ohlcv: pd.DataFrame,
    wins_short=(3, 6, 12, 24, 48),
    micro_wins=(3, 6, 12),
    rsi_windows=(7, 14),
    macd_cfg=(12, 26, 9),
) -> pd.DataFrame:
    """Core OHLCV-derived features. ohlcv has columns: open, high, low, close, volume (uniform freq)."""
    df = pd.DataFrame(index=ohlcv.index)
    ret_1 = logret(ohlcv["close"])
    df["ret_1"] = ret_1

    # rolling stats
    df = df.join(rolling_stats(ret_1, list(wins_short)))
    df = df.join(hl_range(ohlcv["high"], ohlcv["low"], list(wins_short)))
    df = df.join(vol_features(ohlcv["volume"], list(wins_short)))

    # momentum
    for n in rsi_windows:
        df[f"rsi{n}"] = rsi(ohlcv["close"], n)
    m, s, h = macd(ohlcv["close"], *macd_cfg)
    df["macd"] = m
    df["macd_sig"] = s
    df["macd_hist"] = h

    # microstructure
    df = df.join(
        microstructure(
            ohlcv["close"], ohlcv["open"], ret_1, ohlcv["volume"], list(micro_wins)
        )
    )

    # calendar
    df = df.join(add_calendar(ohlcv))

    # clean
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def attach_regime_features(
    features: pd.DataFrame, regime_df: pd.DataFrame | pd.Series, prefix: str = "reg"
) -> pd.DataFrame:
    """
    Align external regime features (e.g., BTC 15m MA, slope) to asset timeframe.
    Regime index must be <= features index (no lookahead). We ffill only.
    """
    if isinstance(regime_df, pd.Series):
        reg = regime_df.rename(f"{prefix}_state").to_frame()
    else:
        reg = regime_df.add_prefix(f"{prefix}_")
    reg = reg.reindex(features.index, method="ffill")
    return features.join(reg)


def make_features_5m_15m(
    ohlcv_5m: pd.DataFrame, regime_15m: pd.Series | pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Build features for 5m bars, with optional BTC 15m regime merged (ffill-aligned).
    """
    feats = make_features_core(ohlcv_5m)
    if regime_15m is not None:
        feats = attach_regime_features(feats, regime_15m, prefix="reg")
    # drop rows with early NaNs from long windows
    return feats.dropna()
