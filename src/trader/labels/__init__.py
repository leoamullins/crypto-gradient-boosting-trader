import numpy as np
import pandas as pd


def add_cost_aware_label(
    df: pd.DataFrame, H: int, roundtrip_cost_logret: float
) -> pd.DataFrame:
    """Binary label: 1 if fwd log-return over H bars beats round-trip costs."""
    out = df.copy()
    fwd = np.log(out["close"].shift(-H)) - np.log(out["close"])
    out["fwd_ret"] = fwd
    out["y"] = (fwd > roundtrip_cost_logret).astype(int)
    return out
