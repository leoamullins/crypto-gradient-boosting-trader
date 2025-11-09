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
) -> pd.Series:
    """
    Walk-forward cross-validation with hold-out test set.

    Args:
        X: Feature DataFrame
        y: Target Series
        embargo: Number of bars to embargo between train/test
        params: Model parameters (if None, uses defaults)
        n_splits: Number of CV splits (increased from 6 to 20)
        mode: 'regression' or 'classification'
        test_size: Fraction of data to hold out for final testing

    Returns:
        Series of predictions (probabilities for classification, returns for regression)
    """
    # Hold out final test set
    n = len(X)
    test_start = int(n * (1 - test_size))
    X_train, y_train = X.iloc[:test_start], y.iloc[:test_start]
    X_test = X.iloc[test_start:]

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

    # Walk-forward CV on training set
    for tr_idx, te_idx in time_series_splits(n_train, n_splits):
        tr_idx = apply_embargo(tr_idx, te_idx, embargo)
        if len(tr_idx) < 500:  # avoid tiny folds
            continue

        mdl = ModelClass(**params)
        mdl.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])

        if mode == "regression":
            preds.iloc[te_idx] = mdl.predict(X_train.iloc[te_idx])
        elif mode == "classification":
            preds.iloc[te_idx] = mdl.predict_proba(X_train.iloc[te_idx])[:, 1]

    # Train final model on all training data for test set
    final_mdl = ModelClass(**params)
    final_mdl.fit(X_train, y_train)

    if mode == "regression":
        preds.iloc[test_start:] = final_mdl.predict(X_test)
    elif mode == "classification":
        preds.iloc[test_start:] = final_mdl.predict_proba(X_test)[:, 1]

    return preds


# Backward compatibility alias
def walkforward_proba(
    X: pd.DataFrame,
    y: pd.Series,
    embargo: int = 48,
    params: dict | None = None,
    n_splits: int = 20,
) -> pd.Series:
    """Legacy function - use walkforward_predict with mode='classification' instead."""
    return walkforward_predict(
        X, y, embargo, params, n_splits, mode="classification", test_size=0.2
    )
