import numpy as np
import pandas as pd

EPS = 1e-12


def logret(series: pd.Series) -> pd.Series:
    return np.log(series).diff()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).rolling(n, min_periods=n).mean()
    dn = (-d.clip(upper=0)).rolling(n, min_periods=n).mean()
    rs = up / (dn + EPS)
    return 100 - 100 / (1 + rs)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ = ema(close, fast)
    slow_ = ema(close, slow)
    macd_ = fast_ - slow_
    signal_ = ema(macd_, signal)
    hist_ = macd_ - signal_
    return macd_, signal_, hist_


def rolling_stats(ret: pd.Series, wins: list[int]) -> pd.DataFrame:
    out = {}
    for w in wins:
        out[f"rmean_{w}"] = ret.rolling(w, min_periods=w).mean()
        out[f"rstd_{w}"] = ret.rolling(w, min_periods=w).std()
    return pd.DataFrame(out, index=ret.index)


def hl_range(high: pd.Series, low: pd.Series, wins: list[int]) -> pd.DataFrame:
    out = {}
    rng = np.log(high / low).replace([np.inf, -np.inf], np.nan)
    for w in wins:
        out[f"range_{w}"] = rng.rolling(w, min_periods=w).mean()
    return pd.DataFrame(out, index=high.index)


def vol_features(volume: pd.Series, wins: list[int]) -> pd.DataFrame:
    return pd.concat(
        {f"vmean_{w}": volume.rolling(w, min_periods=w).mean() for w in wins}, axis=1
    )


def microstructure(
    close: pd.Series,
    open_: pd.Series,
    ret_1: pd.Series,
    volume: pd.Series,
    wins: list[int],
) -> pd.DataFrame:
    upbar = (close > open_).astype(int)
    pieces = {}
    for w in wins:
        pieces[f"up_ratio_{w}"] = upbar.rolling(w, min_periods=w).mean()
        pieces[f"signed_vol_{w}"] = (
            (np.sign(ret_1).fillna(0) * volume).rolling(w, min_periods=w).mean()
        )
    return pd.DataFrame(pieces, index=close.index)
