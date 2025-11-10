import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from trader.backtester.cv import time_series_splits, apply_embargo


def walkforward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    embargo: int = 48,
    params: dict | None = None,
    n_splits: int = 20,
    mode: str = "regression",
    test_size: float = 0.2,
    retrain_freq: int = 288 * 7,  # Retrain weekly (288 5-min bars/day)
) -> pd.Series:
    """Walk-forward CV with embargo and anchored retraining on test set."""
    n = len(X)
    test_start = int(n * (1 - test_size))
    X_train, y_train = X.iloc[:test_start], y.iloc[:test_start]

    if mode == "regression":
        params = params or dict(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            tree_method="hist",
            random_state=42,
        )
        ModelClass = XGBRegressor
    elif mode == "classification":
        params = params or dict(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            tree_method="hist",
            eval_metric="logloss",
            random_state=42,
        )
        ModelClass = XGBClassifier
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'regression' or 'classification'")

    preds = pd.Series(index=X.index, dtype=float)
    n_train = len(X_train)

    # CV predictions on training set
    for tr_idx, te_idx in time_series_splits(n_train, n_splits):
        tr_idx = apply_embargo(tr_idx, te_idx, embargo)
        if len(tr_idx) < 500:
            continue

        mdl = ModelClass(**params)
        mdl.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])

        if mode == "regression":
            preds.iloc[te_idx] = mdl.predict(X_train.iloc[te_idx])
        elif mode == "classification":
            preds.iloc[te_idx] = mdl.predict_proba(X_train.iloc[te_idx])[:, 1]

    # Anchored walk-forward on test set with periodic retraining
    i = test_start
    while i < n:
        # Train on all data up to (i - embargo)
        train_end = i - embargo
        if train_end < 500:
            i += retrain_freq
            continue

        mdl = ModelClass(**params)
        mdl.fit(X.iloc[:train_end], y.iloc[:train_end])

        # Predict next retrain_freq samples (or until end)
        end_idx = min(i + retrain_freq, n)

        if mode == "regression":
            preds.iloc[i:end_idx] = mdl.predict(X.iloc[i:end_idx])
        elif mode == "classification":
            preds.iloc[i:end_idx] = mdl.predict_proba(X.iloc[i:end_idx])[:, 1]

        i += retrain_freq

    return preds


# Backward compatibility alias
def walkforward_proba(
    X: pd.DataFrame,
    y: pd.Series,
    embargo: int = 48,
    params: dict | None = None,
    n_splits: int = 20,
) -> pd.Series:
    """Backward compatibility - use walkforward_predict instead."""
    return walkforward_predict(
        X, y, embargo, params, n_splits, mode="classification", test_size=0.2
    )
