"""
Prediction script for making stock price forecasts.
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
from src.utils.logging_config import setup_logging
from src.utils.visualization import StockVisualizer

logger = logging.getLogger(__name__)


def predict_stock(ticker: str, model_path: str, days_ahead: int = 1, config_path: str = None):
    """
    Make stock price predictions using a trained model.
    
    Args:
        ticker: Stock ticker symbol
        model_path: Path to saved model
        days_ahead: Number of days to predict ahead
        config_path: Path to configuration file
    """
    # Load configuration
    config = get_config(config_path)
    
    # Setup logging
    setup_logging(log_level='INFO')
    
    logger.info("="*80)
    logger.info(f"Making predictions for {ticker}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Prediction horizon: {days_ahead} days")
    logger.info("="*80)
    
    # Initialize components
    data_loader = StockDataLoader(config.project_root)
    feature_engineer = FeatureEngineer()
    preprocessor = DataPreprocessor()
    visualizer = StockVisualizer(save_dir=config.get_path('visualization.plots_dir'))
    
    # Load data
    logger.info("Loading data...")
    df = data_loader.load_individual_stock(
        ticker,
        config.get_path('data.individual_stocks_dir')
    )
    
    # Add features
    logger.info("Engineering features...")
    df = feature_engineer.add_all_features(df, config.feature_config)
    df = preprocessor.handle_missing_values(df)
    
    # Get recent data for prediction
    sequence_length = config.get('data.sequence_length', 60)
    recent_data = df.tail(sequence_length + 100).copy()  # Extra buffer for feature calculation
    
    # Scale features
    feature_cols = preprocessor.get_feature_importance_columns(recent_data)
    recent_scaled, _, _ = preprocessor.scale_features(recent_data, feature_cols=feature_cols)
    
    # Create sequence
    X, _ = preprocessor.create_sequences(
        recent_scaled.tail(sequence_length + days_ahead),
        sequence_length,
        target_col='close',
        feature_cols=feature_cols,
        horizon=days_ahead
    )
    
    # Take the last sequence
    X_pred = X[-1:] if len(X) > 0 else X[:1]
    
    logger.info(f"Input sequence shape: {X_pred.shape}")
    
    # Load model
    logger.info("Loading model...")
    model_path = Path(model_path)
    
    if model_path.suffix == '.keras':
        # Deep learning model
        if 'lstm' in model_path.stem:
            model = LSTMPredictor()
        elif 'gru' in model_path.stem:
            model = GRUPredictor()
        else:
            raise ValueError("Cannot determine model type from filename")
        
        model.load_model(model_path)
    else:
        raise ValueError("Unsupported model format")
    
    # Make prediction
    logger.info("Making prediction...")
    prediction = model.predict(X_pred)[0]
    
    # Get current price
    current_price = df['close'].iloc[-1]
    current_date = df['date'].iloc[-1]
    
    # Calculate prediction metrics
    predicted_change = prediction - current_price
    predicted_change_pct = (predicted_change / current_price) * 100
    
    logger.info("="*80)
    logger.info("PREDICTION RESULTS")
    logger.info("="*80)
    logger.info(f"Stock: {ticker}")
    logger.info(f"Current Date: {current_date}")
    logger.info(f"Current Price: ${current_price:.2f}")
    logger.info(f"Predicted Price ({days_ahead} days ahead): ${prediction:.2f}")
    logger.info(f"Predicted Change: ${predicted_change:.2f} ({predicted_change_pct:+.2f}%)")
    logger.info(f"Direction: {'UP' if predicted_change > 0 else 'DOWN'}")
    logger.info("="*80)
    
    # Create prediction report
    report = pd.DataFrame({
        'Ticker': [ticker],
        'Current Date': [current_date],
        'Current Price': [f'${current_price:.2f}'],
        'Predicted Price': [f'${prediction:.2f}'],
        'Change': [f'${predicted_change:.2f}'],
        'Change %': [f'{predicted_change_pct:+.2f}%'],
        'Direction': ['UP' if predicted_change > 0 else 'DOWN']
    })
    
    # Save report
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / f'{ticker}_prediction_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
    report.to_csv(report_path, index=False)
    logger.info(f"Prediction report saved to {report_path}")
    
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make stock price predictions")
    parser.add_argument('--ticker', type=str, required=True,
                       help='Stock ticker symbol')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--days', type=int, default=1,
                       help='Number of days ahead to predict')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    predict_stock(args.ticker, args.model, args.days, args.config)
