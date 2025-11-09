import numpy as np


def time_series_splits(n: int, n_splits: int = 6):
    """Simple expanding-window split indices for a dataset of length n."""
    fold_sizes = np.linspace(
        0.5, 0.95, n_splits
    )  # last ~5–50% used as tests across folds
    for frac in fold_sizes:
        te_end = int(n * frac)
        te_start = int(te_end - n * 0.10)  # ~10% of data per test fold
        yield np.arange(0, te_start), np.arange(te_start, te_end)


def apply_embargo(tr_idx: np.ndarray, te_idx: np.ndarray, embargo: int) -> np.ndarray:
    """Drop training samples within 'embargo' bars before test start."""
    te_start = te_idx[0]
    return tr_idx[tr_idx < (te_start - embargo)]
