"""
Central configuration for the CMAPSS predictive-maintenance pipeline.

Anything that was a magic number, or a Colab-only path, in the original
notebook lives here so every other module (and eventually the FastAPI
backend) reads it from one place instead of redefining it.
"""

import os

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Training data location — TRAINING-ONLY, never used on the serving path.
# In the original notebook this was a Google Colab Drive mount:
#     from google.colab import drive
#     drive.mount('/content/drive')
#     DATA_DIR = '/content/drive/MyDrive/CMAPSSData/'
# Replaced with a local path / environment variable so it runs anywhere.
# ---------------------------------------------------------------------------
DATA_DIR = os.getenv("CMAPSS_DATA_DIR", "./data/CMAPSSData/")
TRAIN_FILE = "train_FD001.txt"

# ---------------------------------------------------------------------------
# Feature engineering hyperparameters (lifted out of feature_engineering.py
# so they're tunable from one place)
# ---------------------------------------------------------------------------
ROLL_WINDOW = 8      # cycles — short enough to react, long enough to smooth noise
NORM_THRESH = 0.02   # sensors with normalized variance below this are dropped as near-constant

# ---------------------------------------------------------------------------
# RUL construction
# ---------------------------------------------------------------------------
RUL_CLIP = 125

# ---------------------------------------------------------------------------
# Raw CMAPSS column names
# ---------------------------------------------------------------------------
COL_NAMES = (
    ['unit_number', 'time_in_cycles',
     'op_setting_1', 'op_setting_2', 'op_setting_3']
    + [f'sensor_{i:02d}' for i in range(1, 22)]
)
OP_COLS = ['op_setting_1', 'op_setting_2', 'op_setting_3']

# ---------------------------------------------------------------------------
# Engine-level split sizes (FD001 has 100 engines: 80 / 10 / 10)
# ---------------------------------------------------------------------------
N_TRAIN_ENGINES = 80
N_VAL_ENGINES = 10

# ---------------------------------------------------------------------------
# Where trained artifacts (model, scaler, feature lists) get persisted so the
# FastAPI backend can load them without retraining on every request.
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = os.getenv("CMAPSS_ARTIFACTS_DIR", "./artifacts/")

# Notebook-style plots only — the backend never reads from here
OUTPUTS_DIR = "outputs"
