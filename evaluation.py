"""
Metric scorecard shared by every model, plus the results table used to
compare them.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true, y_pred, train_time: float, name: str) -> dict:
    """Compute MAE, RMSE, R2 and store training time — same scorecard for every model."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f'{name:<20} MAE={mae:6.2f}   RMSE={rmse:6.2f}   R2={r2:6.3f}   time={train_time:.3f}s')
    return {'model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'train_time_s': train_time}


def build_results_table(results: list) -> pd.DataFrame:
    return pd.DataFrame(results).sort_values('MAE').reset_index(drop=True)
