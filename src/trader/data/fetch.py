import time
import datetime as dt
import pandas as pd
import numpy as np
import ccxt
from typing import Optional


def timeframe_to_ms(tf: str) -> int:
    unit = tf[-1]
    n = int(tf[:-1])
    mult = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[unit]
    return n * mult


def fetch_ohlcv_all(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str = "5m",
    since: Optional[dt.datetime] = None,
    until: Optional[dt.datetime] = None,
    limit: int = 1000,
    pause: float = 0.2,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    if not exchange.has.get("fetchOHLCV", False):
        raise ValueError(f"{exchange.id} does not support fetchOHLCV")

    tf_ms = timeframe_to_ms(timeframe)
    now_ms = int(time.time() * 1000)
    start_ms = int(since.timestamp() * 1000) if since else now_ms - tf_ms
    end_ms = int(until.timestamp() * 1000) if until else now_ms

    out = []
    last_ms = start_ms
    attempts = 0
    while last_ms < end_ms:
        try:
            batch = exchange.fetch_ohlcv(
                symbol=symbol, timeframe=timeframe, since=last_ms, limit=limit
            )
            if not batch:
                break
            out.extend(batch)
            last_ms = batch[-1][0] + tf_ms
            if pause:
                time.sleep(pause)
            attempts = 0
            if max_rows and len(out) >= max_rows:
                break
        except (ccxt.NetworkError, ccxt.RateLimitExceeded) as e:
            time.sleep(1.5)
            attempts += 1
            if attempts > 5:
                raise RuntimeError(f"too many retries: {e}")
        except ccxt.ExchangeError as e:
            raise

    if not out:
        raise RuntimeError(f"No OHLCV data returned for {symbol}")

    df = pd.DataFrame(
        out, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index().drop_duplicates()
    return df
