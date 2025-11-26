"""
Main entry point for the stock prediction system.
"""

import argparse
import logging
from pathlib import Path

from src.config import get_config
from src.utils.logging_config import setup_logging
from train import train_model
from predict import predict_stock

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Stock Analysis and Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train LSTM model on AAPL
  python main.py train --ticker AAPL --model lstm
  
  # Train ensemble model on all stocks
  python main.py train --model ensemble
  
  # Make prediction for AAPL
  python main.py predict --ticker AAPL --model models/saved_models/AAPL_lstm_model.keras
  
  # Make 5-day ahead prediction
  python main.py predict --ticker AAPL --model models/saved_models/AAPL_lstm_model.keras --days 5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a prediction model')
    train_parser.add_argument('--ticker', type=str, default=None,
                            help='Stock ticker symbol (default: all stocks)')
    train_parser.add_argument('--model', type=str, default='lstm',
                            choices=['lstm', 'gru', 'random_forest', 'ensemble'],
                            help='Model type to train')
    train_parser.add_argument('--config', type=str, default=None,
                            help='Path to configuration file')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--ticker', type=str, required=True,
                              help='Stock ticker symbol')
    predict_parser.add_argument('--model', type=str, required=True,
                              help='Path to trained model')
    predict_parser.add_argument('--days', type=int, default=1,
                              help='Number of days ahead to predict')
    predict_parser.add_argument('--config', type=str, default=None,
                              help='Path to configuration file')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging(log_level='INFO')
    
    if args.command == 'train':
        logger.info("Starting training process...")
        train_model(ticker=args.ticker, model_type=args.model, config_path=args.config)
    
    elif args.command == 'predict':
        logger.info("Starting prediction process...")
        predict_stock(args.ticker, args.model, args.days, args.config)


if __name__ == "__main__":
    main()
