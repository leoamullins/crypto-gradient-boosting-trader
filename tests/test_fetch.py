from trader.data import make_exchange, fetch_ohlcv_all, save_parquet
from trader.strategy.regime import compute_regime
import datetime as dt, pytz

tz = pytz.UTC

ex = make_exchange("binance")
start = dt.datetime(2025, 1, 1, tzinfo=tz)
df = fetch_ohlcv_all(ex, "BTC/USDT", timeframe="5m", since=start)
df2 = fetch_ohlcv_all(ex, "BTC/USDT", timeframe="15m", since=start)

regime = compute_regime(df2)

save_parquet(regime.to_frame(), "raw/BTCUSDT_15m_regime.parquet")
save_parquet(df, "raw/BTCUSDT_5m.parquet")
print(df.tail())
