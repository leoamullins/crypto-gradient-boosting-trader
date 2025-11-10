import numpy as np
import pandas as pd


def add_cost_aware_label(
    df: pd.DataFrame, H: int, roundtrip_cost_logret: float, mode: str = "regression"
) -> pd.DataFrame:
    """Add forward return labels with transaction costs."""
    out = df.copy()
    fwd = np.log(out["close"].shift(-H)) - np.log(out["close"])
    out["fwd_ret"] = fwd

    if mode == "regression":
        # Regression target: net return after costs
        out["y"] = fwd - roundtrip_cost_logret
    elif mode == "binary":
        # Binary target: 1 if beats cost, 0 otherwise
        out["y"] = (fwd > roundtrip_cost_logret).astype(int)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'regression' or 'binary'")

    return out
