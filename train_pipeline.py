"""
Offline training script. Run this once (or whenever you retrain) — it is
NEVER called by the FastAPI backend at request time. It reproduces the full
notebook top-to-bottom and persists everything inference.py needs.

Usage:
    python train_pipeline.py
"""

import os
import json
import joblib

from config import DATA_DIR, TRAIN_FILE, ARTIFACTS_DIR
from data_loader import build_labeled_dataset, split_by_engine
from feature_engineering import select_informative_sensors, add_rolling_features, get_feature_columns, get_Xy
from preprocessing import fit_scaler, apply_scaler
from models import (
    train_baseline, train_linear_regression, train_decision_tree,
    train_random_forest, train_xgboost, save_model,
)
from evaluation import build_results_table


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 1. Load + label + split (identical seed/proportions as Session 1)
    train_df = build_labeled_dataset(os.path.join(DATA_DIR, TRAIN_FILE))
    df_train, df_val, df_test = split_by_engine(train_df)

    # 2. Feature engineering — decided on TRAIN only
    keep_sensors, drop_sensors = select_informative_sensors(df_train)
    df_train_fe = add_rolling_features(df_train, keep_sensors)
    df_val_fe = add_rolling_features(df_val, keep_sensors)
    df_test_fe = add_rolling_features(df_test, keep_sensors)

    feature_cols = get_feature_columns(keep_sensors)
    X_train, y_train = get_Xy(df_train_fe, feature_cols)
    X_val, y_val = get_Xy(df_val_fe, feature_cols)
    X_test, y_test = get_Xy(df_test_fe, feature_cols)

    # 3. Scale — fit on train only
    scaler = fit_scaler(X_train)
    X_train_scaled = apply_scaler(scaler, X_train)
    X_val_scaled = apply_scaler(scaler, X_val)
    X_test_scaled = apply_scaler(scaler, X_test)

    # 4. Train + evaluate every model
    results = []
    _, baseline_metrics, _ = train_baseline(df_train_fe, df_val_fe)
    results.append(baseline_metrics)

    lin_model, lin_metrics, _ = train_linear_regression(X_train_scaled, y_train, X_val_scaled, y_val)
    results.append(lin_metrics)

    tree_model, tree_metrics, _ = train_decision_tree(X_train_scaled, y_train, X_val_scaled, y_val)
    results.append(tree_metrics)

    rf_model, rf_metrics, _ = train_random_forest(X_train_scaled, y_train, X_val_scaled, y_val)
    results.append(rf_metrics)

    xgb_model, xgb_metrics, _ = train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val)
    results.append(xgb_metrics)

    results_df = build_results_table(results)
    print(results_df)

    # 5. Persist the model used for serving (XGBoost, per the notebook's
    #    example) + everything inference.py needs to replay this exact pipeline
    save_model(xgb_model, os.path.join(ARTIFACTS_DIR, 'model.joblib'))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'scaler.joblib'))

    with open(os.path.join(ARTIFACTS_DIR, 'keep_sensors.json'), 'w') as f:
        json.dump(keep_sensors, f)
    with open(os.path.join(ARTIFACTS_DIR, 'feature_cols.json'), 'w') as f:
        json.dump(feature_cols, f)

    print(f'\nArtifacts saved to {ARTIFACTS_DIR}')


if __name__ == '__main__':
    main()
