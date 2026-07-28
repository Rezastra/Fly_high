"""
Feature scaling. The scaler is FIT ONLY on training data (in train_pipeline.py)
and persisted — at serving time inference.py only ever calls apply_scaler()
(transform, never fit).
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    """Transform only — never refit. Safe to call on train, val, test, or live request data."""
    return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
