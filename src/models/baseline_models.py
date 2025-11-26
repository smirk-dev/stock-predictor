"""
Baseline models for comparison: Random Forest, Linear Regression, XGBoost.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class BaselinePredictor:
    """Baseline machine learning models for stock prediction."""
    
    def __init__(self, model_type: str = 'random_forest', config: dict = None):
        """
        Initialize baseline predictor.
        
        Args:
            model_type: Type of model ('random_forest', 'linear', 'ridge', 'lasso')
            config: Configuration dictionary
        """
        self.model_type = model_type
        self.config = config or {}
        self.model = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the model based on type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=self.config.get('n_estimators', 200),
                max_depth=self.config.get('max_depth', 20),
                min_samples_split=self.config.get('min_samples_split', 5),
                min_samples_leaf=self.config.get('min_samples_leaf', 2),
                random_state=self.config.get('random_seed', 42),
                n_jobs=-1,
                verbose=0
            )
        elif self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=self.config.get('alpha', 1.0))
        elif self.model_type == 'lasso':
            self.model = Lasso(alpha=self.config.get('alpha', 1.0))
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Initialized {self.model_type} model")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the model.
        
        Args:
            X_train: Training features (flattened for non-sequential models)
            y_train: Training targets
        """
        # Flatten 3D sequences to 2D if needed
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)
        
        logger.info(f"Training {self.model_type} model with {X_train.shape[0]} samples...")
        
        self.model.fit(X_train, y_train)
        
        logger.info("Training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Flatten 3D sequences to 2D if needed
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model...")
        
        predictions = self.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions)
        }
        
        # MAPE
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        metrics['mape'] = mape
        
        # Directional accuracy
        y_direction = np.sign(np.diff(y_test, prepend=y_test[0]))
        pred_direction = np.sign(np.diff(predictions, prepend=predictions[0]))
        metrics['directional_accuracy'] = np.mean(y_direction == pred_direction)
        
        logger.info(f"Evaluation results: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")
        
        return metrics
    
    def get_feature_importance(self, feature_names: list = None) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores (only for tree-based models).
        
        Args:
            feature_names: Names of features
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning(f"{self.model_type} does not support feature importance")
            return None
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        importance_dict = dict(zip(feature_names, importances))
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
