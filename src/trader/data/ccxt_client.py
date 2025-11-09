import ccxt
from typing import Optional


def make_exchange(
    exchange_id: str = "binance",
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    rate_limit: bool = True,
):
    klass = getattr(ccxt, exchange_id)
    params = {"enableRateLimit": rate_limit}

    if api_key:
        paramas["apiKey"] = api_key
    if api_secret:
        params["secret"] = api_secret

    exchange = klass(params)
    exchange.load_markets()
    return exchange
