"""
Model training. Each train_* function fits one model on the training set,
scores it on validation via evaluate() (evaluation.py), and returns
(model, metrics_dict, val_predictions) instead of leaving loose notebook
variables (lin_model, lin_time, ...) around.

save_model / load_model persist a trained model so the FastAPI backend can
load it without retraining on every request.
"""

import time
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from config import RANDOM_STATE
from evaluation import evaluate


def train_baseline(df_train_fe, df_val_fe):
    """Always predict the TRAIN mean RUL — never computed on val/test (same anti-leakage rule)."""
    baseline_value = df_train_fe['RUL'].mean()
    baseline_pred_val = np.full(len(df_val_fe), baseline_value)
    metrics = evaluate(df_val_fe['RUL'], baseline_pred_val, 0.0, 'Baseline (mean)')
    return baseline_value, metrics, baseline_pred_val


def train_linear_regression(X_train, y_train, X_val, y_val):
    t0 = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    pred_val = model.predict(X_val)
    metrics = evaluate(y_val, pred_val, train_time, 'Linear Regression')
    return model, metrics, pred_val


def train_decision_tree(X_train, y_train, X_val, y_val, max_depth=6, random_state=RANDOM_STATE):
    t0 = time.time()
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    pred_val = model.predict(X_val)
    metrics = evaluate(y_val, pred_val, train_time, 'Decision Tree')
    return model, metrics, pred_val


def train_random_forest(X_train, y_train, X_val, y_val,
                         n_estimators=300, max_depth=10, random_state=RANDOM_STATE):
    t0 = time.time()
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                   random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    pred_val = model.predict(X_val)
    metrics = evaluate(y_val, pred_val, train_time, 'Random Forest')
    return model, metrics, pred_val


def train_xgboost(X_train, y_train, X_val, y_val,
                   n_estimators=300, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE):
    t0 = time.time()
    model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                          random_state=random_state, n_jobs=-1, objective='reg:squarederror')
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    pred_val = model.predict(X_val)
    metrics = evaluate(y_val, pred_val, train_time, 'XGBoost')
    return model, metrics, pred_val


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
