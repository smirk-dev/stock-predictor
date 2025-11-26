"""
Evaluation metrics for stock prediction models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate various evaluation metrics for stock predictions."""
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                             prefix: str = "") -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            prefix: Prefix for metric names
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Basic regression metrics
        metrics[f'{prefix}rmse'] = MetricsCalculator.rmse(y_true, y_pred)
        metrics[f'{prefix}mae'] = MetricsCalculator.mae(y_true, y_pred)
        metrics[f'{prefix}mape'] = MetricsCalculator.mape(y_true, y_pred)
        metrics[f'{prefix}r2'] = MetricsCalculator.r2(y_true, y_pred)
        
        # Stock-specific metrics
        metrics[f'{prefix}directional_accuracy'] = MetricsCalculator.directional_accuracy(y_true, y_pred)
        metrics[f'{prefix}sharpe_ratio'] = MetricsCalculator.sharpe_ratio(y_true, y_pred)
        
        # Error statistics
        errors = y_true - y_pred
        metrics[f'{prefix}mean_error'] = np.mean(errors)
        metrics[f'{prefix}std_error'] = np.std(errors)
        metrics[f'{prefix}max_error'] = np.max(np.abs(errors))
        
        return metrics
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared score."""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate directional accuracy (percentage of correct direction predictions).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Directional accuracy as percentage
        """
        if len(y_true) < 2:
            return 0.0
        
        true_direction = np.sign(np.diff(y_true, prepend=y_true[0]))
        pred_direction = np.sign(np.diff(y_pred, prepend=y_pred[0]))
        
        accuracy = np.mean(true_direction == pred_direction) * 100
        return accuracy
    
    @staticmethod
    def sharpe_ratio(y_true: np.ndarray, y_pred: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio based on prediction returns.
        
        Args:
            y_true: True prices
            y_pred: Predicted prices
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        # Calculate returns
        true_returns = np.diff(y_true) / y_true[:-1]
        pred_returns = np.diff(y_pred) / y_pred[:-1]
        
        # Strategy returns (go long when predicting up, short when predicting down)
        pred_direction = np.sign(pred_returns)
        strategy_returns = true_returns * pred_direction
        
        if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming daily data with 252 trading days)
        excess_returns = strategy_returns - (risk_free_rate / 252)
        sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(strategy_returns)
        
        return sharpe
    
    @staticmethod
    def maximum_drawdown(prices: np.ndarray) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            prices: Array of prices or portfolio values
            
        Returns:
            Maximum drawdown as percentage
        """
        cumulative_max = np.maximum.accumulate(prices)
        drawdown = (prices - cumulative_max) / cumulative_max
        max_drawdown = np.min(drawdown) * 100
        
        return max_drawdown
    
    @staticmethod
    def hit_rate(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.02) -> float:
        """
        Calculate hit rate (percentage of predictions within threshold of true value).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            threshold: Acceptable error threshold as percentage
            
        Returns:
            Hit rate as percentage
        """
        percentage_errors = np.abs((y_true - y_pred) / y_true)
        hits = np.sum(percentage_errors <= threshold)
        hit_rate = (hits / len(y_true)) * 100
        
        return hit_rate
    
    @staticmethod
    def calculate_returns_metrics(returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate metrics for a returns series.
        
        Args:
            returns: Array of returns
            
        Returns:
            Dictionary with returns metrics
        """
        metrics = {
            'total_return': (np.prod(1 + returns) - 1) * 100,
            'mean_return': np.mean(returns) * 100,
            'std_return': np.std(returns) * 100,
            'max_return': np.max(returns) * 100,
            'min_return': np.min(returns) * 100,
            'positive_days': np.sum(returns > 0) / len(returns) * 100,
            'sharpe_ratio': np.sqrt(252) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        }
        
        return metrics
    
    @staticmethod
    def create_metrics_report(y_true: np.ndarray, y_pred: np.ndarray, 
                            model_name: str = "Model") -> pd.DataFrame:
        """
        Create a formatted metrics report.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            DataFrame with formatted metrics
        """
        metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred)
        
        report_data = []
        for metric_name, value in metrics.items():
            report_data.append({
                'Model': model_name,
                'Metric': metric_name,
                'Value': f"{value:.4f}"
            })
        
        return pd.DataFrame(report_data)
    
    @staticmethod
    def compare_models(y_true: np.ndarray, predictions_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Compare multiple models' predictions.
        
        Args:
            y_true: True values
            predictions_dict: Dictionary mapping model names to predictions
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for model_name, y_pred in predictions_dict.items():
            metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred)
            metrics['model'] = model_name
            comparison_data.append(metrics)
        
        df = pd.DataFrame(comparison_data)
        
        # Reorder columns to have model first
        cols = ['model'] + [col for col in df.columns if col != 'model']
        df = df[cols]
        
        return df
