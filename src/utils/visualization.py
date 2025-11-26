"""
Visualization utilities for stock analysis and predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (15, 8)


class StockVisualizer:
    """Create visualizations for stock data and predictions."""
    
    def __init__(self, save_dir: str = None):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_price_history(self, df: pd.DataFrame, ticker: str = None,
                          save_name: str = None) -> plt.Figure:
        """
        Plot price history with volume.
        
        Args:
            df: DataFrame with price data
            ticker: Stock ticker symbol
            save_name: Filename to save plot
            
        Returns:
            Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Price plot
        ax1.plot(df['date'], df['close'], label='Close Price', linewidth=2)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title(f'{"Stock" if ticker is None else ticker} Price History', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume plot
        ax2.bar(df['date'], df['volume'], alpha=0.7, color='steelblue')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_title('Trading Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {self.save_dir / save_name}")
        
        return fig
    
    def plot_predictions(self, dates: np.ndarray, actual: np.ndarray,
                        predicted: np.ndarray, model_name: str = "Model",
                        save_name: str = None) -> plt.Figure:
        """
        Plot actual vs predicted prices.
        
        Args:
            dates: Array of dates
            actual: Actual prices
            predicted: Predicted prices
            model_name: Name of the model
            save_name: Filename to save plot
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        
        ax.plot(dates, actual, label='Actual', linewidth=2, alpha=0.8)
        ax.plot(dates, predicted, label='Predicted', linewidth=2, alpha=0.8, linestyle='--')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title(f'{model_name} - Actual vs Predicted Prices', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_multiple_predictions(self, dates: np.ndarray, actual: np.ndarray,
                                 predictions_dict: Dict[str, np.ndarray],
                                 save_name: str = None) -> plt.Figure:
        """
        Plot multiple model predictions.
        
        Args:
            dates: Array of dates
            actual: Actual prices
            predictions_dict: Dictionary mapping model names to predictions
            save_name: Filename to save plot
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        
        ax.plot(dates, actual, label='Actual', linewidth=2.5, color='black', alpha=0.8)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (model_name, predicted) in enumerate(predictions_dict.items()):
            ax.plot(dates, predicted, label=model_name, linewidth=2, 
                   alpha=0.7, linestyle='--', color=colors[i % len(colors)])
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title('Model Comparison - Predictions vs Actual', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_technical_indicators(self, df: pd.DataFrame, save_name: str = None) -> plt.Figure:
        """
        Plot technical indicators.
        
        Args:
            df: DataFrame with technical indicators
            save_name: Filename to save plot
            
        Returns:
            Figure object
        """
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        
        # Price with Moving Averages
        axes[0].plot(df['date'], df['close'], label='Close', linewidth=2)
        if 'sma_20' in df.columns:
            axes[0].plot(df['date'], df['sma_20'], label='SMA 20', alpha=0.7)
        if 'sma_50' in df.columns:
            axes[0].plot(df['date'], df['sma_50'], label='SMA 50', alpha=0.7)
        if 'sma_200' in df.columns:
            axes[0].plot(df['date'], df['sma_200'], label='SMA 200', alpha=0.7)
        axes[0].set_ylabel('Price ($)')
        axes[0].set_title('Price with Moving Averages', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RSI
        if 'rsi_14' in df.columns:
            axes[1].plot(df['date'], df['rsi_14'], label='RSI', color='purple', linewidth=2)
            axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
            axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
            axes[1].set_ylabel('RSI')
            axes[1].set_title('Relative Strength Index (RSI)', fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # MACD
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            axes[2].plot(df['date'], df['macd'], label='MACD', linewidth=2)
            axes[2].plot(df['date'], df['macd_signal'], label='Signal', linewidth=2)
            if 'macd_histogram' in df.columns:
                axes[2].bar(df['date'], df['macd_histogram'], label='Histogram', alpha=0.3)
            axes[2].set_ylabel('MACD')
            axes[2].set_title('MACD Indicator', fontweight='bold')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        # Bollinger Bands
        if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
            axes[3].plot(df['date'], df['close'], label='Close', linewidth=2)
            axes[3].plot(df['date'], df['bb_upper'], label='Upper Band', alpha=0.5, linestyle='--')
            axes[3].plot(df['date'], df['bb_middle'], label='Middle Band', alpha=0.5)
            axes[3].plot(df['date'], df['bb_lower'], label='Lower Band', alpha=0.5, linestyle='--')
            axes[3].fill_between(df['date'], df['bb_lower'], df['bb_upper'], alpha=0.1)
            axes[3].set_xlabel('Date')
            axes[3].set_ylabel('Price ($)')
            axes[3].set_title('Bollinger Bands', fontweight='bold')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_training_history(self, history: dict, save_name: str = None) -> plt.Figure:
        """
        Plot model training history.
        
        Args:
            history: Training history dictionary
            save_name: Filename to save plot
            
        Returns:
            Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE
        if 'mean_absolute_error' in history:
            axes[1].plot(history['mean_absolute_error'], label='Training MAE', linewidth=2)
            if 'val_mean_absolute_error' in history:
                axes[1].plot(history['val_mean_absolute_error'], label='Validation MAE', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].set_title('Mean Absolute Error', fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                               save_name: str = None) -> plt.Figure:
        """
        Plot prediction error distribution.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_name: Filename to save plot
            
        Returns:
            Figure object
        """
        errors = y_true - y_pred
        percentage_errors = (errors / y_true) * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Absolute errors
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Prediction Error ($)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Prediction Errors', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Percentage errors
        axes[1].hist(percentage_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Prediction Error (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Percentage Errors', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self, importance_dict: Dict[str, float],
                               top_n: int = 20, save_name: str = None) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            importance_dict: Dictionary mapping feature names to importance scores
            top_n: Number of top features to display
            save_name: Filename to save plot
            
        Returns:
            Figure object
        """
        # Get top features
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importances = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, alpha=0.8, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Feature Importances', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame, columns: List[str] = None,
                               save_name: str = None) -> plt.Figure:
        """
        Plot correlation matrix heatmap.
        
        Args:
            df: DataFrame
            columns: Columns to include in correlation matrix
            save_name: Filename to save plot
            
        Returns:
            Figure object
        """
        if columns is None:
            # Use numeric columns
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        corr_matrix = df[columns].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_portfolio_performance(self, portfolio_values: List[float],
                                  dates: np.ndarray = None,
                                  save_name: str = None) -> plt.Figure:
        """
        Plot portfolio value over time.
        
        Args:
            portfolio_values: List of portfolio values
            dates: Optional array of dates
            save_name: Filename to save plot
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        
        x_values = dates if dates is not None else range(len(portfolio_values))
        
        ax.plot(x_values, portfolio_values, linewidth=2, color='green')
        ax.fill_between(x_values, portfolio_values, alpha=0.3, color='green')
        
        ax.set_xlabel('Date' if dates is not None else 'Time Period')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Portfolio Performance Over Time', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line for initial value
        ax.axhline(y=portfolio_values[0], color='r', linestyle='--', 
                  alpha=0.5, label=f'Initial Value: ${portfolio_values[0]:,.2f}')
        ax.legend()
        
        plt.tight_layout()
        
        if save_name and self.save_dir:
            plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        
        return fig
