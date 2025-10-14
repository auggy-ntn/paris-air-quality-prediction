
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')



def hyperparameter_tuning(X_train, y_train, n_trials=50, cv_splits=3, test_size=504):
    """
    Perform hyperparameter tuning using RandomizedSearchCV with TimeSeriesSplit
    
    Args:
        X_train: Training features
        y_train: Training targets
        n_trials: Number of random parameter combinations to try
        cv_splits: Number of cross-validation splits
        test_size: Size of test set for each CV split
        
    Returns:
        dict: Best parameters and best score
    """
    print("Starting hyperparameter tuning...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0]
    }
    
    # Create TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=cv_splits, test_size=test_size)
    
    # Initialize XGBoost model
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    
    # Perform randomized search
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=n_trials,
        cv=tscv,
        scoring='mean_absolute_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    random_search.fit(X_train, y_train)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best CV score: {-random_search.best_score_:.4f}")
    
    return {
        'best_params': random_search.best_params_,
        'best_score': -random_search.best_score_,
        'best_model': random_search.best_estimator_
    }


def predict(model, X_test):
    """
    Make predictions using trained model
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        
    Returns:
        np.ndarray: Predictions
    """
    return model.predict(X_test)

def train_model(X_train, y_train, params=None):
    """Train an XGBoost model with optional parameters"""
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42}
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model