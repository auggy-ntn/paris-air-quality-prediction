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
from src.utils.feature_engineering import add_lags


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


def predict(models, X_test:pd.DataFrame, X_past:pd.DataFrame, lags) -> np.ndarray:
    """
    Make predictions using trained model
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        
    Returns:
        np.ndarray: Predictions
    """
    full_data = pd.concat([X_past, X_test])
    for line in X_test.index:
        full_data = add_lags(full_data, lags=lags)
        X_input = full_data.loc[line]
        for col, model in models.items():
            y_pred = model.predict(X_input)
            full_data.loc[line, col] = y_pred[0]
    return full_data.loc[X_test.index]

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

class XGBoostModels:
    def __init__(self, lags, target_columns, params=None):
        self.lags = lags
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        self.target_columns = target_columns
        self.params = {col: params.copy() for col in target_columns}
        self.models = {col: xgb.XGBRegressor(**params) for col in target_columns}
        self.columns = None
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame, tune_hyperparameters=False):
        """Fit the model to training data"""
        self.columns = X_train.columns.tolist()
        for col in y_train.columns:
            print(f"Training model for {col}...")
            y_col = y_train[col]
            X_col = X_train.copy()

            model = self.models[col]
            model.fit(X_col, y_col)
            self.models[col] = model
            print(f"Model trained for {col}.")
            
        return self
        
    def predict_single(self, X: pd.Series) -> pd.DataFrame:
        """Make predictions on test data"""
        preds = pd.DataFrame(columns=self.target_columns)
        for col, model in self.models.items():
            preds.loc[X.name, col] = model.predict(X.values.reshape(1, -1))
        return preds
    
    def predict(self, X_test:pd.DataFrame, X_past:pd.DataFrame) -> pd.DataFrame:
        """Make predictions on test data"""
        full_data = pd.concat([X_past, X_test])
        for line in X_test.index:
            full_data = add_lags(full_data, lags=self.lags)
            X_input = full_data.loc[line][self.columns]
            prediction = self.predict_single(X_input)
            for col in self.target_columns:
                full_data.loc[line, col] = prediction.loc[line, col]
        return full_data.loc[X_test.index]
    
    def hyperparameter_tuning(self, X_train, y_train, n_trials=50, cv_splits=3, test_size=504):
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
        self.columns = X_train.columns.tolist()
        print("Starting hyperparameter tuning...")
        
        for col in self.target_columns:
            print(f"Tuning hyperparameters for {col}...")
            y_col = y_train[col]
            X_col = X_train.copy()
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
            xgb_model = xgb.XGBRegressor(
                random_state=42, 
                n_jobs=-1
            )
            
            # Perform randomized search
            random_search = RandomizedSearchCV(
                estimator=xgb_model,
                param_distributions=param_grid,
                n_iter=n_trials,
                cv=tscv,
                scoring='neg_mean_absolute_error',
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
            
            # Fit the model
            random_search.fit(X_col, y_col)
            
            print(f"Best parameters: {random_search.best_params_}")
            print(f"Best CV score: {random_search.best_score_:.4f}")
            self.params[col] = random_search.best_params_
            self.models[col] = random_search.best_estimator_

        return self