"""
Main training script for stock prediction models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging

from src.config import get_config
from src.data.data_loader import StockDataLoader
from src.data.feature_engineering import FeatureEngineer
from src.data.preprocessing import DataPreprocessor
from src.models.lstm_model import LSTMPredictor
from src.models.gru_model import GRUPredictor
from src.models.baseline_models import BaselinePredictor
from src.models.ensemble_model import EnsemblePredictor
from src.evaluation.metrics import MetricsCalculator
from src.utils.logging_config import setup_logging
from src.utils.visualization import StockVisualizer

logger = logging.getLogger(__name__)


def train_model(ticker: str = None, model_type: str = 'lstm', config_path: str = None):
    """
    Train a stock prediction model.
    
    Args:
        ticker: Stock ticker symbol (if None, uses all stocks)
        model_type: Type of model to train ('lstm', 'gru', 'random_forest', 'ensemble')
        config_path: Path to configuration file
    """
    # Load configuration
    config = get_config(config_path)
    
    # Setup logging
    setup_logging(
        log_level=config.get('logging.level', 'INFO'),
        log_file=config.get('logging.log_file', 'logs/training.log')
    )
    
    logger.info("="*80)
    logger.info(f"Starting training for {'All Stocks' if ticker is None else ticker}")
    logger.info(f"Model Type: {model_type}")
    logger.info("="*80)
    
    # Initialize components
    data_loader = StockDataLoader(config.project_root)
    feature_engineer = FeatureEngineer()
    preprocessor = DataPreprocessor(scaler_type='standard')
    visualizer = StockVisualizer(save_dir=config.get_path('visualization.plots_dir'))
    
    # Load data
    logger.info("Loading data...")
    if ticker:
        df = data_loader.load_individual_stock(
            ticker,
            config.get_path('data.individual_stocks_dir')
        )
    else:
        df = data_loader.load_all_stocks(config.get_path('data.all_stocks_path'))
    
    logger.info(f"Loaded {len(df)} records")
    
    # Add technical indicators
    logger.info("Engineering features...")
    df = feature_engineer.add_all_features(df, config.feature_config)
    
    # Handle missing values
    df = preprocessor.handle_missing_values(df)
    
    # Add target variable
    horizon = config.get('prediction.horizons', [1])[0]
    df = preprocessor.add_target_variable(df, horizon=horizon)
    
    # Remove rows with NaN target
    df = df.dropna(subset=['target'])
    
    logger.info(f"Dataset shape after preprocessing: {df.shape}")
    
    # Split data
    logger.info("Splitting data...")
    splits = preprocessor.split_data(
        df,
        train_ratio=config.get('data.train_split'),
        val_ratio=config.get('data.validation_split'),
        test_ratio=config.get('data.test_split'),
        by_ticker='name' in df.columns
    )
    
    train_df, val_df, test_df = splits['train'], splits['val'], splits['test']
    
    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    
    # Scale features
    logger.info("Scaling features...")
    feature_cols = preprocessor.get_feature_importance_columns(train_df)
    train_scaled, val_scaled, test_scaled = preprocessor.scale_features(
        train_df, val_df, test_df, feature_cols
    )
    
    # Create sequences for deep learning models
    if model_type in ['lstm', 'gru', 'ensemble']:
        logger.info("Creating sequences...")
        sequence_length = config.get('data.sequence_length', 60)
        
        X_train, y_train = preprocessor.create_sequences(
            train_scaled, sequence_length, target_col='target', feature_cols=feature_cols
        )
        X_val, y_val = preprocessor.create_sequences(
            val_scaled, sequence_length, target_col='target', feature_cols=feature_cols
        )
        X_test, y_test = preprocessor.create_sequences(
            test_scaled, sequence_length, target_col='target', feature_cols=feature_cols
        )
    else:
        # For baseline models, use flat features
        X_train = train_scaled[feature_cols].values
        y_train = train_scaled['target'].values
        X_val = val_scaled[feature_cols].values
        y_val = val_scaled['target'].values
        X_test = test_scaled[feature_cols].values
        y_test = test_scaled['target'].values
    
    logger.info(f"Training data shape: {X_train.shape}")
    
    # Train model
    logger.info(f"Training {model_type} model...")
    checkpoint_dir = config.get_path('training.checkpoint_dir')
    
    if model_type == 'lstm':
        model = LSTMPredictor(config.model_config.get('lstm', {}))
        history = model.train(X_train, y_train, X_val, y_val, checkpoint_dir=checkpoint_dir)
        
        # Plot training history
        visualizer.plot_training_history(history, save_name=f'{ticker or "all"}_lstm_history.png')
        
    elif model_type == 'gru':
        model = GRUPredictor(config.model_config.get('gru', {}))
        history = model.train(X_train, y_train, X_val, y_val, checkpoint_dir=checkpoint_dir)
        
        # Plot training history
        visualizer.plot_training_history(history, save_name=f'{ticker or "all"}_gru_history.png')
        
    elif model_type == 'random_forest':
        model = BaselinePredictor('random_forest', config.model_config.get('random_forest', {}))
        model.train(X_train, y_train)
        
    elif model_type == 'ensemble':
        logger.info("Training ensemble of models...")
        
        # Train LSTM
        lstm = LSTMPredictor(config.model_config.get('lstm', {}))
        lstm.train(X_train, y_train, X_val, y_val, checkpoint_dir=checkpoint_dir)
        
        # Train GRU
        gru = GRUPredictor(config.model_config.get('gru', {}))
        gru.train(X_train, y_train, X_val, y_val, checkpoint_dir=checkpoint_dir)
        
        # Train Random Forest
        rf = BaselinePredictor('random_forest', config.model_config.get('random_forest', {}))
        rf.train(X_train, y_train)
        
        # Create ensemble
        weights = config.model_config.get('ensemble', {}).get('weights', [0.4, 0.3, 0.3])
        model = EnsemblePredictor([lstm, gru, rf], weights=weights, 
                                 model_names=['LSTM', 'GRU', 'RandomForest'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Evaluate on test set
    logger.info("Evaluating model...")
    
    if model_type == 'ensemble':
        results = model.evaluate(X_test, y_test, return_individual=True)
        logger.info("\nEnsemble Metrics:")
        for metric, value in results['ensemble'].items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Individual model metrics
        logger.info("\nIndividual Model Metrics:")
        for model_name, metrics in results['individual'].items():
            logger.info(f"  {model_name}:")
            for metric, value in metrics.items():
                logger.info(f"    {metric}: {value:.4f}")
    else:
        metrics = model.evaluate(X_test, y_test)
        logger.info("\nTest Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    # Make predictions for visualization
    logger.info("Generating predictions...")
    predictions = model.predict(X_test)
    
    # Get dates for test set
    if len(test_scaled) > len(y_test):
        test_dates = test_scaled['date'].iloc[-len(y_test):].values
    else:
        test_dates = test_scaled['date'].values
    
    # Plot predictions
    visualizer.plot_predictions(
        test_dates, y_test, predictions,
        model_name=model_type.upper(),
        save_name=f'{ticker or "all"}_{model_type}_predictions.png'
    )
    
    # Plot error distribution
    visualizer.plot_error_distribution(
        y_test, predictions,
        save_name=f'{ticker or "all"}_{model_type}_errors.png'
    )
    
    # Save model
    save_dir = config.get_path('training.model_save_dir')
    model_path = save_dir / f"{ticker or 'all'}_{model_type}_model.keras"
    
    if hasattr(model, 'save_model'):
        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
    
    logger.info("="*80)
    logger.info("Training completed successfully!")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stock prediction model")
    parser.add_argument('--ticker', type=str, default=None, 
                       help='Stock ticker symbol (default: all stocks)')
    parser.add_argument('--model', type=str, default='lstm',
                       choices=['lstm', 'gru', 'random_forest', 'ensemble'],
                       help='Model type to train')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    train_model(ticker=args.ticker, model_type=args.model, config_path=args.config)
