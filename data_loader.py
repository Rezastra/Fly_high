"""
Data loading.

Two independent entry points that never call each other:

  TRAINING PATH (used only by train_pipeline.py):
      load_cmapss           -> read one raw CMAPSS .txt file
      build_labeled_dataset -> attach the RUL target
      split_by_engine       -> engine-level train/val/test split

  SERVING PATH (used only by inference.py, called from the FastAPI backend):
      records_to_dataframe    -> turn a JSON payload (list of dicts) into a
                                  DataFrame shaped like the raw CMAPSS data
      dataset_file_to_dataframe -> turn an uploaded dataset FILE (the
                                  user picks a file on their PC) into the
                                  same shape
"""

import io

import numpy as np
import pandas as pd

from config import COL_NAMES, RUL_CLIP, RANDOM_STATE, N_TRAIN_ENGINES, N_VAL_ENGINES


# ---------------------------------------------------------------------------
# TRAINING PATH
# ---------------------------------------------------------------------------

def load_cmapss(filepath: str) -> pd.DataFrame:
    """Read one raw, whitespace-separated CMAPSS file into a labeled-columns DataFrame."""
    df = pd.read_csv(filepath, sep=r'\s+', header=None, engine='python')
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = COL_NAMES
    return df


def build_labeled_dataset(filepath: str) -> pd.DataFrame:
    """Load a training file and attach the RUL target (clipped, same as Session 1)."""
    train_df = load_cmapss(filepath)

    engine_lives = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    engine_lives.rename(columns={'time_in_cycles': 'max_cycles'}, inplace=True)
    train_df = train_df.merge(engine_lives, on='unit_number')
    train_df['RUL_raw'] = train_df['max_cycles'] - train_df['time_in_cycles']
    train_df['RUL'] = train_df['RUL_raw'].clip(upper=RUL_CLIP)

    print(f'Loaded {train_df.shape[0]:,} rows across {train_df["unit_number"].nunique()} engines')
    return train_df


def split_by_engine(df: pd.DataFrame, random_state: int = RANDOM_STATE,
                     n_train: int = N_TRAIN_ENGINES, n_val: int = N_VAL_ENGINES):
    """Engine-level split — identical seed/proportions as Session 1. Never split by row."""
    np.random.seed(random_state)
    all_engines = df['unit_number'].unique()
    np.random.shuffle(all_engines)

    train_engines = all_engines[:n_train]
    val_engines = all_engines[n_train:n_train + n_val]
    test_engines = all_engines[n_train + n_val:]

    df_train = df[df['unit_number'].isin(train_engines)].copy()
    df_val = df[df['unit_number'].isin(val_engines)].copy()
    df_test = df[df['unit_number'].isin(test_engines)].copy()

    assert set(train_engines) & set(val_engines) == set(), 'Leakage detected!'
    assert set(train_engines) & set(test_engines) == set(), 'Leakage detected!'

    print(f'Train: {len(train_engines)} engines — {len(df_train):,} rows')
    print(f'Val:   {len(val_engines)} engines — {len(df_val):,} rows')
    print(f'Test:  {len(test_engines)} engines — {len(df_test):,} rows')
    print('✓ No leakage — every engine belongs to exactly one set.')

    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# SERVING PATH
# ---------------------------------------------------------------------------

def records_to_dataframe(records: list) -> pd.DataFrame:
    """
    Convert a list of JSON records posted by the frontend into a DataFrame
    shaped like the raw CMAPSS data (minus RUL — that's what we're predicting).

    Each record is expected to be a dict with keys:
        unit_number, time_in_cycles, op_setting_1..3, sensor_01..21

    Multiple rows per unit_number are expected: enough consecutive cycles
    (>= ROLL_WINDOW, see config.py) to compute rolling features for that engine.
    """
    expected_cols = list(COL_NAMES)
    df = pd.DataFrame.from_records(records)

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Incoming data is missing required columns: {missing}')

    df = df[expected_cols].sort_values(['unit_number', 'time_in_cycles']).reset_index(drop=True)
    return df


def dataset_file_to_dataframe(raw_bytes: bytes, filename: str = '') -> pd.DataFrame:
    """
    Parse a dataset FILE the user selected on their own PC and uploaded
    through the frontend, into a DataFrame shaped like the raw CMAPSS data.

    Accepts either:
      - the raw NASA CMAPSS format (whitespace-separated, no header — same
        as train_FD001.txt / test_FD001.txt), or
      - a .csv with a header row using the same column names as COL_NAMES.

    Does NOT require (or use) a RUL column — RUL is what the model predicts.
    """
    text = raw_bytes.decode('utf-8', errors='ignore')
    if not text.strip():
        raise ValueError('Uploaded file is empty.')

    first_line = text.strip().splitlines()[0]
    looks_like_header = ('unit_number' in first_line) or ('sensor_01' in first_line)

    if looks_like_header or filename.lower().endswith('.csv'):
        df = pd.read_csv(io.StringIO(text))
        if 'unit_number' not in df.columns:
            # Header row present but didn't match expected names — fall back
            # to treating the whole file as headerless whitespace data.
            df = pd.read_csv(io.StringIO(text), sep=r'\s+', header=None, engine='python')
            df.dropna(axis=1, how='all', inplace=True)
    else:
        df = pd.read_csv(io.StringIO(text), sep=r'\s+', header=None, engine='python')
        df.dropna(axis=1, how='all', inplace=True)

    if df.shape[1] == len(COL_NAMES) and not looks_like_header:
        df.columns = COL_NAMES
    elif 'unit_number' not in df.columns:
        raise ValueError(
            f'Could not match the uploaded file to the expected {len(COL_NAMES)} '
            f'CMAPSS columns (found {df.shape[1]}). Expected either the raw '
            f'whitespace-separated NASA format, or a .csv with headers: {COL_NAMES}'
        )

    missing = [c for c in COL_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f'Uploaded file is missing required columns: {missing}')

    df = df[COL_NAMES].sort_values(['unit_number', 'time_in_cycles']).reset_index(drop=True)
    return df
