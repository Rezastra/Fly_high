"""
Feature importance — two independent methods, kept numeric-only here.
Plotting (if you want the charts) lives in visualization.py.
"""

import pandas as pd
from sklearn.inspection import permutation_importance

from config import RANDOM_STATE


def get_builtin_importances(model, feature_cols: list) -> pd.Series:
    """Tree-based models only (e.g. Random Forest)."""
    return pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)


def get_permutation_importances(model, X_val, y_val, feature_cols: list,
                                 n_repeats: int = 10, random_state: int = RANDOM_STATE) -> pd.Series:
    perm = permutation_importance(model, X_val, y_val, n_repeats=n_repeats,
                                   random_state=random_state, scoring='neg_mean_absolute_error')
    return pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)
