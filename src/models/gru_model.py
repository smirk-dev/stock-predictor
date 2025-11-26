"""
GRU model for stock price prediction.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from typing import Tuple, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class GRUPredictor:
    """GRU-based stock price predictor."""
    
    def __init__(self, config: dict = None):
        """
        Initialize GRU predictor.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or {}
        self.model = None
        self.history = None
        
        # Default hyperparameters
        self.units = self.config.get('units', [128, 64])
        self.dropout = self.config.get('dropout', 0.2)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 32)
        self.epochs = self.config.get('epochs', 100)
        self.patience = self.config.get('patience', 15)
        
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build GRU model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building GRU model with input shape: {input_shape}")
        
        model = keras.Sequential()
        
        # First GRU layer with return sequences
        model.add(layers.GRU(
            units=self.units[0],
            return_sequences=True if len(self.units) > 1 else False,
            input_shape=input_shape
        ))
        model.add(layers.Dropout(self.dropout))
        
        # Additional GRU layers
        for i, units in enumerate(self.units[1:], 1):
            return_seq = i < len(self.units) - 1
            model.add(layers.GRU(units=units, return_sequences=return_seq))
            model.add(layers.Dropout(self.dropout))
        
        # Dense layers for output
        model.add(layers.Dense(units=32, activation='relu'))
        model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(units=1))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mean_absolute_error', 'mean_absolute_percentage_error']
        )
        
        self.model = model
        
        logger.info(f"Model built successfully with {model.count_params()} parameters")
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray = None, y_val: np.ndarray = None,
             checkpoint_dir: str = None) -> dict:
        """
        Train the GRU model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            checkpoint_dir: Directory to save model checkpoints
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)
        
        logger.info(f"Training GRU model for {self.epochs} epochs...")
        
        # Setup callbacks
        callback_list = self._setup_callbacks(checkpoint_dir)
        
        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=1
        )
        
        logger.info("Training completed")
        
        return self.history.history
    
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
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("Evaluating model...")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        metrics = {
            'loss': results[0],
            'mae': results[1],
            'mape': results[2]
        }
        
        # Additional metrics
        predictions = self.predict(X_test)
        
        from sklearn.metrics import mean_squared_error, r2_score
        metrics['rmse'] = np.sqrt(mean_squared_error(y_test, predictions))
        metrics['r2'] = r2_score(y_test, predictions)
        
        # Directional accuracy
        y_direction = np.sign(np.diff(y_test, prepend=y_test[0]))
        pred_direction = np.sign(np.diff(predictions, prepend=predictions[0]))
        metrics['directional_accuracy'] = np.mean(y_direction == pred_direction)
        
        logger.info(f"Evaluation results: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")
        
        return metrics
    
    def _setup_callbacks(self, checkpoint_dir: str = None) -> List[callbacks.Callback]:
        """Setup training callbacks."""
        callback_list = []
        
        # Early stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss' if self.history is None else 'loss',
            patience=self.patience,
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stop)
        
        # Model checkpoint
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = checkpoint_dir / "gru_best.keras"
            model_checkpoint = callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_loss' if self.history is None else 'loss',
                save_best_only=True,
                verbose=1
            )
            callback_list.append(model_checkpoint)
        
        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if self.history is None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        return callback_list
    
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
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built"
        
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)
