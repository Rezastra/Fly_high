"""
Feature engineering.

select_informative_sensors: TRAINING-ONLY — decides which sensors to keep.
    On the serving path, reuse the persisted `keep_sensors` list from
    training. Never recompute it against a live/incoming batch.

add_rolling_features: shared by training and serving.

get_feature_columns / get_Xy: assemble the final feature matrix.
"""

import pandas as pd

from config import ROLL_WINDOW, NORM_THRESH, OP_COLS


def select_informative_sensors(df_train: pd.DataFrame, thresh: float = NORM_THRESH):
    """Decide, using TRAIN ONLY, which sensors carry real signal vs. near-constant noise."""
    sensor_cols = [c for c in df_train.columns if c.startswith('sensor_')]

    sensor_range = df_train[sensor_cols].max() - df_train[sensor_cols].min()
    non_const = sensor_range[sensor_range > 1e-9].index.tolist()

    mins, maxs = df_train[non_const].min(), df_train[non_const].max()
    norm = (df_train[non_const] - mins) / (maxs - mins)
    var_norm = norm.var().sort_values()

    keep_sensors = var_norm[var_norm >= thresh].index.tolist()
    drop_sensors = var_norm[var_norm < thresh].index.tolist()

    print(f'Keeping {len(keep_sensors)} informative sensors')
    print(f'Dropping {len(drop_sensors)} near-constant sensors: {drop_sensors}')
    return keep_sensors, drop_sensors


def add_rolling_features(df: pd.DataFrame, sensors: list, window: int = ROLL_WINDOW) -> pd.DataFrame:
    """Add a rolling mean/std per sensor, computed within each engine only."""
    df = df.sort_values(['unit_number', 'time_in_cycles']).copy()
    for s in sensors:
        grouped = df.groupby('unit_number')[s]
        df[f'{s}_rmean'] = grouped.transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'{s}_rstd'] = grouped.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
    return df


def get_feature_columns(keep_sensors: list, op_cols: list = OP_COLS) -> list:
    """Final feature list: operating settings + raw sensors + their rolling stats."""
    rolling_cols = [f'{s}_rmean' for s in keep_sensors] + [f'{s}_rstd' for s in keep_sensors]
    return op_cols + keep_sensors + rolling_cols


def get_Xy(df_fe: pd.DataFrame, feature_cols: list, target_col: str = 'RUL'):
    """Slice out X (and y, if the target column is present — it won't be at serving time)."""
    X = df_fe[feature_cols]
    y = df_fe[target_col] if target_col in df_fe.columns else None
    return X, y
