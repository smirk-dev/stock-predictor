"""
Simple example: Train a model and make predictions on Apple stock.
This is a minimal example to get started quickly.
"""

import logging
from src.config import get_config
from src.data.data_loader import StockDataLoader
from src.data.feature_engineering import FeatureEngineer
from src.data.preprocessing import DataPreprocessor
from src.models.lstm_model import LSTMPredictor
from src.utils.logging_config import setup_logging
from src.utils.visualization import StockVisualizer

# Setup logging
setup_logging(log_level='INFO')
logger = logging.getLogger(__name__)

def main():
    """Simple example workflow."""
    
    print("\n" + "="*80)
    print("SIMPLE STOCK PREDICTION EXAMPLE - AAPL")
    print("="*80 + "\n")
    
    # 1. Load configuration
    logger.info("Step 1: Loading configuration...")
    config = get_config()
    
    # 2. Load data
    logger.info("Step 2: Loading AAPL stock data...")
    loader = StockDataLoader(config.project_root)
    df = loader.load_individual_stock('AAPL', config.get_path('data.individual_stocks_dir'))
    logger.info(f"Loaded {len(df)} records from {df['date'].min()} to {df['date'].max()}")
    
    # 3. Add technical indicators
    logger.info("Step 3: Adding technical indicators...")
    engineer = FeatureEngineer()
    df = engineer.add_all_features(df, config.feature_config)
    logger.info(f"Dataset now has {len(df.columns)} columns")
    
    # 4. Preprocess data
    logger.info("Step 4: Preprocessing data...")
    preprocessor = DataPreprocessor()
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.add_target_variable(df, horizon=1)
    df = df.dropna(subset=['target'])
    
    # 5. Split data
    logger.info("Step 5: Splitting into train/val/test sets...")
    splits = preprocessor.split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    train_df, val_df, test_df = splits['train'], splits['val'], splits['test']
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 6. Scale features
    logger.info("Step 6: Scaling features...")
    feature_cols = preprocessor.get_feature_importance_columns(train_df)
    train_scaled, val_scaled, test_scaled = preprocessor.scale_features(
        train_df, val_df, test_df, feature_cols
    )
    
    # 7. Create sequences
    logger.info("Step 7: Creating sequences for LSTM...")
    sequence_length = 60
    X_train, y_train = preprocessor.create_sequences(train_scaled, sequence_length, 
                                                     target_col='target', feature_cols=feature_cols)
    X_val, y_val = preprocessor.create_sequences(val_scaled, sequence_length,
                                                 target_col='target', feature_cols=feature_cols)
    X_test, y_test = preprocessor.create_sequences(test_scaled, sequence_length,
                                                   target_col='target', feature_cols=feature_cols)
    logger.info(f"Training sequences shape: {X_train.shape}")
    
    # 8. Train LSTM model
    logger.info("Step 8: Training LSTM model (this may take a few minutes)...")
    model = LSTMPredictor({
        'units': [128, 64],
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,  # Reduced for quick demo
        'patience': 10
    })
    
    history = model.train(X_train, y_train, X_val, y_val)
    
    # 9. Evaluate model
    logger.info("Step 9: Evaluating model on test set...")
    metrics = model.evaluate(X_test, y_test)
    
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper():.<40} {value:.4f}")
    print("="*80)
    
    # 10. Make predictions
    logger.info("Step 10: Making predictions...")
    predictions = model.predict(X_test)
    
    # 11. Visualize results
    logger.info("Step 11: Creating visualizations...")
    visualizer = StockVisualizer(save_dir='outputs/plots')
    
    # Get dates for test set
    test_dates = test_scaled['date'].iloc[-len(y_test):].values
    
    # Plot predictions
    visualizer.plot_predictions(test_dates, y_test, predictions, 
                               model_name='LSTM', save_name='example_predictions.png')
    
    # Plot training history
    visualizer.plot_training_history(history, save_name='example_training.png')
    
    # Plot errors
    visualizer.plot_error_distribution(y_test, predictions, save_name='example_errors.png')
    
    logger.info("Visualizations saved to outputs/plots/")
    
    # 12. Make a prediction for tomorrow
    logger.info("Step 12: Predicting tomorrow's price...")
    last_sequence = X_test[-1:]
    tomorrow_prediction = model.predict(last_sequence)[0]
    current_price = df['close'].iloc[-1]
    
    print("\n" + "="*80)
    print("TOMORROW'S PREDICTION")
    print("="*80)
    print(f"Current Price (last known): ${current_price:.2f}")
    print(f"Predicted Price (tomorrow): ${tomorrow_prediction:.2f}")
    print(f"Expected Change: ${tomorrow_prediction - current_price:.2f} "
          f"({((tomorrow_prediction - current_price) / current_price * 100):+.2f}%)")
    print(f"Direction: {'📈 UP' if tomorrow_prediction > current_price else '📉 DOWN'}")
    print("="*80 + "\n")
    
    logger.info("✓ Example completed successfully!")
    logger.info("Check outputs/plots/ for visualizations")
    logger.info("\nNext steps:")
    logger.info("  - Try different models: GRU, Random Forest, Ensemble")
    logger.info("  - Experiment with hyperparameters in config.yaml")
    logger.info("  - Explore notebooks for detailed analysis")
    logger.info("  - Run: python main.py train --ticker AAPL --model lstm")
    
    print()


if __name__ == "__main__":
    main()
