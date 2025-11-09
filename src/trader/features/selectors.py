from typing import List

EXCLUDE_COLS = {"y", "fwd_ret"}  # labels added later in labels module


def default_feature_list(columns: List[str]) -> List[str]:
    return [c for c in columns if c not in EXCLUDE_COLS]


def safe_float_cols(df):
    # ensure boosters get numeric dtype
    return df.select_dtypes(include=["number"]).columns.tolist()
