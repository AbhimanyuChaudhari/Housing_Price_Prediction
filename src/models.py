"""
models.py
---------
Model definitions, cross-validation utilities, stacking ensemble,
and Optuna hyperparameter tuning for the Ames Housing regression task.
"""

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor,
    StackingRegressor
)
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# ── CV utility ────────────────────────────────────────────────────────────────

def cv_rmse(model, X, y, n_splits=5, random_state=42):
    """Return array of RMSE scores from k-fold CV (log-space target)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        model, X, y, cv=kf,
        scoring='neg_root_mean_squared_error'
    )
    return -scores


def evaluate_all(X, y, verbose=True):
    """
    Run CV RMSE for all baseline models.
    Returns a DataFrame sorted by mean RMSE.
    """
    models = get_baseline_models()
    rows = []
    for name, model in models.items():
        scores = cv_rmse(model, X, y)
        rows.append({
            'Model': name,
            'CV RMSE Mean': scores.mean(),
            'CV RMSE Std':  scores.std(),
            'CV R² (approx)': 1 - (scores.mean() / np.std(y))**2,
        })
        if verbose:
            print(f'  {name:30s}  RMSE: {scores.mean():.5f} ± {scores.std():.5f}')
    return pd.DataFrame(rows).sort_values('CV RMSE Mean').reset_index(drop=True)


# ── Baseline models ───────────────────────────────────────────────────────────

def get_baseline_models():
    return {
        'Ridge':              Ridge(alpha=10),
        'Lasso':              Lasso(alpha=0.0005, max_iter=10000),
        'ElasticNet':         ElasticNet(alpha=0.0005, l1_ratio=0.9, max_iter=10000),
        'SVR':                SVR(C=20, epsilon=0.008, gamma=0.0003, kernel='rbf'),
        'Random Forest':      RandomForestRegressor(
                                  n_estimators=300, max_depth=None,
                                  min_samples_split=5, random_state=42, n_jobs=-1),
        'Gradient Boosting':  GradientBoostingRegressor(
                                  n_estimators=300, learning_rate=0.05,
                                  max_depth=4, min_samples_leaf=15,
                                  loss='huber', random_state=42),
        'XGBoost':            xgb.XGBRegressor(
                                  n_estimators=300, learning_rate=0.05,
                                  max_depth=4, colsample_bytree=0.8,
                                  subsample=0.8, reg_alpha=0.1,
                                  reg_lambda=1.0, random_state=42,
                                  verbosity=0),
        'LightGBM':           lgb.LGBMRegressor(
                                  n_estimators=300, learning_rate=0.05,
                                  num_leaves=31, colsample_bytree=0.8,
                                  subsample=0.8, reg_alpha=0.1,
                                  reg_lambda=1.0, random_state=42,
                                  verbose=-1),
    }


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def tune_xgboost(X, y, n_trials=50):
    """Use Optuna to find optimal XGBoost hyperparameters."""

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 200, 600),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth':        trial.suggest_int('max_depth', 3, 6),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': 42, 'verbosity': 0,
        }
        model = xgb.XGBRegressor(**params)
        scores = cv_rmse(model, X, y, n_splits=5)
        return scores.mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f'\nBest XGBoost RMSE: {study.best_value:.5f}')
    print(f'Best params: {study.best_params}')
    return study.best_params


def tune_lightgbm(X, y, n_trials=50):
    """Use Optuna to find optimal LightGBM hyperparameters."""

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 200, 600),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves':       trial.suggest_int('num_leaves', 20, 80),
            'max_depth':        trial.suggest_int('max_depth', 3, 8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'min_child_samples':trial.suggest_int('min_child_samples', 5, 50),
            'random_state': 42, 'verbose': -1,
        }
        model = lgb.LGBMRegressor(**params)
        scores = cv_rmse(model, X, y, n_splits=5)
        return scores.mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f'\nBest LightGBM RMSE: {study.best_value:.5f}')
    print(f'Best params: {study.best_params}')
    return study.best_params


# ── Stacking ensemble ─────────────────────────────────────────────────────────

def build_stacking_ensemble(xgb_params=None, lgb_params=None):
    """
    Two-level stacking ensemble.

    Level 1 (base learners): Ridge, Lasso, GBM, XGBoost, LightGBM
    Level 2 (meta learner):  Ridge (low complexity to avoid overfitting)

    Using cross_val_predict internally ensures no data leakage.
    """
    xgb_params = xgb_params or {}
    lgb_params  = lgb_params  or {}

    base_learners = [
        ('ridge',   Ridge(alpha=10)),
        ('lasso',   Lasso(alpha=0.0005, max_iter=10000)),
        ('gbm',     GradientBoostingRegressor(
                        n_estimators=300, learning_rate=0.05,
                        max_depth=4, loss='huber', random_state=42)),
        ('xgb',     xgb.XGBRegressor(
                        **{**{'n_estimators': 300, 'learning_rate': 0.05,
                              'max_depth': 4, 'random_state': 42, 'verbosity': 0},
                           **xgb_params})),
        ('lgbm',    lgb.LGBMRegressor(
                        **{**{'n_estimators': 300, 'learning_rate': 0.05,
                              'num_leaves': 31, 'random_state': 42, 'verbose': -1},
                           **lgb_params})),
    ]

    stack = StackingRegressor(
        estimators=base_learners,
        final_estimator=Ridge(alpha=10),
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        passthrough=True,   # also pass original features to meta learner
        n_jobs=-1,
    )
    return stack


# ── Weighted average blend ────────────────────────────────────────────────────

def weighted_blend(models_preds: dict, weights: dict = None):
    """
    Simple weighted average of model predictions (in original price space).
    Falls back to equal weights if none provided.
    """
    if weights is None:
        weights = {k: 1.0 / len(models_preds) for k in models_preds}
    total_w = sum(weights.values())
    blended = sum(
        models_preds[name] * (w / total_w)
        for name, w in weights.items()
        if name in models_preds
    )
    return blended
