"""
Notebook / report plots only. The FastAPI backend never imports this file —
a frontend renders its own charts from the JSON the API returns. Kept here
so you (or an instructor deck) can still regenerate the same figures.
"""

import matplotlib.pyplot as plt
import pandas as pd

from config import OUTPUTS_DIR


def set_plot_style():
    plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False,
                          'axes.spines.right': False, 'font.size': 12})


def plot_rolling_example(df_train_fe: pd.DataFrame, sensor_to_show: str, roll_window: int, example_engine):
    sub = df_train_fe[df_train_fe['unit_number'] == example_engine]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sub['time_in_cycles'], sub[sensor_to_show], color='#2E75B6', alpha=0.5, lw=1.2, label='raw value')
    ax.plot(sub['time_in_cycles'], sub[f'{sensor_to_show}_rmean'], color='#2E8B57', lw=2.4,
            label=f'rolling mean ({roll_window} cycles)')
    ax.set_xlabel('Cycle'); ax.set_ylabel(sensor_to_show)
    ax.set_title(f'Engine {example_engine} — {sensor_to_show}: raw vs. smoothed trend')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}/rolling_feature_example.png', bbox_inches='tight')
    plt.show()


def plot_model_comparison(results_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ['#7F8C8D', '#2E75B6', '#2ECC71', '#145A32', '#E67E22']
    ax.bar(results_df['model'], results_df['MAE'], color=colors[:len(results_df)])
    ax.set_ylabel('MAE on validation (cycles)')
    ax.set_title('Model comparison — lower is better')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}/model_comparison_mae.png', bbox_inches='tight')
    plt.show()


def plot_residuals(y_val, best_preds, best_name: str):
    residuals = y_val.values - best_preds
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(best_preds, residuals, alpha=0.3, s=12, color='#2E75B6')
    ax.axhline(0, color='#C0392B', lw=1.5, ls='--')
    ax.set_xlabel('Predicted RUL'); ax.set_ylabel('Residual (true − predicted)')
    ax.set_title(f'Residuals — {best_name} (validation set)')
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}/residuals_best_model.png', bbox_inches='tight')
    plt.show()


def plot_builtin_importance(importances: pd.Series, top_n: int = 10):
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.head(top_n).sort_values().plot(kind='barh', ax=ax, color='#2E8B57')
    ax.set_title(f'Random Forest — top {top_n} feature importances')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}/feature_importance.png', bbox_inches='tight')
    plt.show()


def plot_permutation_importance(perm_importances: pd.Series, top_n: int = 10):
    fig, ax = plt.subplots(figsize=(8, 5))
    perm_importances.head(top_n).sort_values().plot(kind='barh', ax=ax, color='#1F4E79')
    ax.set_title(f'Permutation importance — top {top_n} (validation set)')
    ax.set_xlabel('Increase in MAE when shuffled')
    plt.tight_layout()
    plt.savefig(f'{OUTPUTS_DIR}/permutation_importance.png', bbox_inches='tight')
    plt.show()
