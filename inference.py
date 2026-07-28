"""
Serving path. This is the ONLY module your FastAPI backend needs to import
for a /predict-style endpoint.

load_artifacts() loads everything training decided (model, scaler,
keep_sensors, feature_cols) once — e.g. at API startup.
run_inference() takes the frontend's raw records and returns a score per
engine.
"""

import os
import json
import joblib

from config import ARTIFACTS_DIR, ROLL_WINDOW
from data_loader import records_to_dataframe
from feature_engineering import add_rolling_features, get_Xy
from preprocessing import apply_scaler
from risk import rul_to_risk, recommendation


def load_artifacts(artifacts_dir: str = ARTIFACTS_DIR) -> dict:
    """Load everything the training pipeline persisted. Call once, e.g. at API startup."""
    model = joblib.load(os.path.join(artifacts_dir, 'model.joblib'))
    scaler = joblib.load(os.path.join(artifacts_dir, 'scaler.joblib'))

    with open(os.path.join(artifacts_dir, 'keep_sensors.json')) as f:
        keep_sensors = json.load(f)
    with open(os.path.join(artifacts_dir, 'feature_cols.json')) as f:
        feature_cols = json.load(f)

    return {
        'model': model,
        'scaler': scaler,
        'keep_sensors': keep_sensors,
        'feature_cols': feature_cols,
    }


def run_inference(records: list, artifacts: dict) -> list:
    """
    records:   list of dicts posted by the frontend — multiple rows per
               unit_number, at least ROLL_WINDOW cycles of history each.
    artifacts: the dict returned by load_artifacts().

    Returns one result per engine present in `records`, scored on each
    engine's MOST RECENT cycle.
    """
    df = records_to_dataframe(records)
    df_fe = add_rolling_features(df, artifacts['keep_sensors'], window=ROLL_WINDOW)

    X, _ = get_Xy(df_fe, artifacts['feature_cols'])
    X_scaled = apply_scaler(artifacts['scaler'], X)

    preds = artifacts['model'].predict(X_scaled)
    df_fe = df_fe.assign(predicted_rul=preds)

    # one prediction per engine: its latest cycle
    latest = df_fe.sort_values('time_in_cycles').groupby('unit_number').tail(1)

    results = []
    for _, row in latest.iterrows():
        risk = rul_to_risk(row['predicted_rul'])
        results.append({
            'engine_id': int(row['unit_number']),
            'predicted_rul': round(float(row['predicted_rul']), 1),
            'risk': risk,
            'recommendation': recommendation(risk),
        })
    return results


def score_dataset(df, artifacts: dict) -> list:
    """
    Score a whole uploaded dataset — multiple engines, full cycle histories
    (as opposed to run_inference(), which only scores each engine's latest
    cycle). Used by the /api/predict-dataset endpoint.

    Returns one entry per engine: its latest predicted_rul/risk/recommendation
    plus a cycle-by-cycle history (for a frontend trend chart).
    """
    df_fe = add_rolling_features(df, artifacts['keep_sensors'], window=ROLL_WINDOW)

    X, _ = get_Xy(df_fe, artifacts['feature_cols'])
    X_scaled = apply_scaler(artifacts['scaler'], X)

    preds = artifacts['model'].predict(X_scaled)
    df_fe = df_fe.assign(predicted_rul=preds)

    results = []
    for engine_id, engine_df in df_fe.sort_values('time_in_cycles').groupby('unit_number'):
        history = [
            {'cycle': int(r['time_in_cycles']), 'predicted_rul': round(float(r['predicted_rul']), 1)}
            for _, r in engine_df.iterrows()
        ]
        latest = engine_df.iloc[-1]
        risk = rul_to_risk(latest['predicted_rul'])
        results.append({
            'engine_id': int(engine_id),
            'current_cycle': int(latest['time_in_cycles']),
            'predicted_rul': round(float(latest['predicted_rul']), 1),
            'risk': risk,
            'recommendation': recommendation(risk),
            'history': history,
        })
    return results
