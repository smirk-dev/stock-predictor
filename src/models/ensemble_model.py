"""
Ensemble model that combines predictions from multiple models.
"""

import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Ensemble model combining multiple predictors."""
    
    def __init__(self, models: List, weights: Optional[List[float]] = None, 
                 model_names: Optional[List[str]] = None):
        """
        Initialize ensemble predictor.
        
        Args:
            models: List of trained model objects
            weights: Weights for each model (must sum to 1.0)
            model_names: Names for each model
        """
        self.models = models
        
        # Default equal weights
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        if len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")
        
        self.weights = np.array(weights)
        
        if model_names is None:
            model_names = [f"model_{i}" for i in range(len(models))]
        
        self.model_names = model_names
        
        logger.info(f"Initialized ensemble with {len(models)} models")
        logger.info(f"Weights: {dict(zip(model_names, weights))}")
    
    def predict(self, X: np.ndarray, return_individual: bool = False) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Input features
            return_individual: If True, return individual model predictions
            
        Returns:
            Ensemble predictions (or dict with individual predictions if return_individual=True)
        """
        logger.info("Making ensemble predictions...")
        
        # Get predictions from each model
        individual_predictions = []
        
        for i, model in enumerate(self.models):
            try:
                pred = model.predict(X)
                individual_predictions.append(pred)
                logger.info(f"{self.model_names[i]} prediction completed")
            except Exception as e:
                logger.error(f"Error with {self.model_names[i]}: {str(e)}")
                raise
        
        # Convert to numpy array
        individual_predictions = np.array(individual_predictions)
        
        # Weighted average
        ensemble_pred = np.average(individual_predictions, axis=0, weights=self.weights)
        
        if return_individual:
            return {
                'ensemble': ensemble_pred,
                'individual': {
                    name: pred for name, pred in zip(self.model_names, individual_predictions)
                }
            }
        
        return ensemble_pred
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                return_individual: bool = True) -> Dict:
        """
        Evaluate ensemble performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            return_individual: If True, include individual model metrics
            
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        logger.info("Evaluating ensemble...")
        
        # Get predictions
        pred_results = self.predict(X_test, return_individual=True)
        ensemble_pred = pred_results['ensemble']
        
        # Calculate ensemble metrics
        ensemble_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
            'mae': mean_absolute_error(y_test, ensemble_pred),
            'r2': r2_score(y_test, ensemble_pred),
            'mape': np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
        }
        
        # Directional accuracy
        y_direction = np.sign(np.diff(y_test, prepend=y_test[0]))
        pred_direction = np.sign(np.diff(ensemble_pred, prepend=ensemble_pred[0]))
        ensemble_metrics['directional_accuracy'] = np.mean(y_direction == pred_direction)
        
        results = {'ensemble': ensemble_metrics}
        
        # Individual model metrics
        if return_individual:
            individual_metrics = {}
            
            for name, pred in pred_results['individual'].items():
                metrics = {
                    'rmse': np.sqrt(mean_squared_error(y_test, pred)),
                    'mae': mean_absolute_error(y_test, pred),
                    'r2': r2_score(y_test, pred)
                }
                
                pred_dir = np.sign(np.diff(pred, prepend=pred[0]))
                metrics['directional_accuracy'] = np.mean(y_direction == pred_dir)
                
                individual_metrics[name] = metrics
            
            results['individual'] = individual_metrics
        
        logger.info(f"Ensemble RMSE: {ensemble_metrics['rmse']:.4f}, MAE: {ensemble_metrics['mae']:.4f}, R2: {ensemble_metrics['r2']:.4f}")
        
        return results
    
    def optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray, 
                        method: str = 'grid_search') -> np.ndarray:
        """
        Optimize ensemble weights based on validation performance.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            method: Optimization method ('grid_search' or 'minimize')
            
        Returns:
            Optimized weights
        """
        from sklearn.metrics import mean_squared_error
        
        logger.info(f"Optimizing ensemble weights using {method}...")
        
        # Get individual predictions on validation set
        individual_predictions = []
        for model in self.models:
            pred = model.predict(X_val)
            individual_predictions.append(pred)
        
        individual_predictions = np.array(individual_predictions)
        
        if method == 'grid_search':
            # Grid search over possible weight combinations
            best_weights = None
            best_score = float('inf')
            
            # Generate weight combinations
            weight_options = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
            
            from itertools import product
            
            for weights in product(weight_options, repeat=len(self.models)):
                weights = np.array(weights)
                
                # Ensure weights sum to 1
                if abs(np.sum(weights) - 1.0) < 1e-6:
                    # Calculate ensemble prediction
                    ensemble_pred = np.average(individual_predictions, axis=0, weights=weights)
                    
                    # Calculate error
                    mse = mean_squared_error(y_val, ensemble_pred)
                    
                    if mse < best_score:
                        best_score = mse
                        best_weights = weights
            
            self.weights = best_weights
            logger.info(f"Optimized weights: {dict(zip(self.model_names, best_weights))}")
            logger.info(f"Validation MSE: {best_score:.4f}")
        
        elif method == 'minimize':
            from scipy.optimize import minimize
            
            def objective(weights):
                weights = weights / np.sum(weights)  # Normalize
                ensemble_pred = np.average(individual_predictions, axis=0, weights=weights)
                return mean_squared_error(y_val, ensemble_pred)
            
            # Initial guess (equal weights)
            x0 = np.ones(len(self.models)) / len(self.models)
            
            # Constraints: weights sum to 1 and are non-negative
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
            bounds = [(0, 1) for _ in range(len(self.models))]
            
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            self.weights = result.x
            logger.info(f"Optimized weights: {dict(zip(self.model_names, self.weights))}")
            logger.info(f"Validation MSE: {result.fun:.4f}")
        
        return self.weights
