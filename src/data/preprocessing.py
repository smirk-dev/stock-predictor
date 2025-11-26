"""
Data preprocessing and preparation for model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess and prepare stock data for machine learning models."""
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
        """
        self.scaler_type = scaler_type
        self.feature_scaler = None
        self.target_scaler = None
        self._initialize_scalers()
        
    def _initialize_scalers(self):
        """Initialize the scalers based on type."""
        scalers = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler
        }
        
        if self.scaler_type not in scalers:
            logger.warning(f"Unknown scaler type: {self.scaler_type}. Using StandardScaler.")
            self.scaler_type = 'standard'
        
        self.feature_scaler = scalers[self.scaler_type]()
        self.target_scaler = scalers[self.scaler_type]()
    
    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.7, 
                   val_ratio: float = 0.15, test_ratio: float = 0.15,
                   by_ticker: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            by_ticker: If True, split each ticker separately to avoid data leakage
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        if by_ticker and 'name' in df.columns:
            logger.info("Splitting data by ticker to avoid data leakage")
            train_dfs, val_dfs, test_dfs = [], [], []
            
            for ticker in df['name'].unique():
                ticker_df = df[df['name'] == ticker].copy()
                ticker_splits = self._split_single_series(ticker_df, train_ratio, val_ratio, test_ratio)
                train_dfs.append(ticker_splits['train'])
                val_dfs.append(ticker_splits['val'])
                test_dfs.append(ticker_splits['test'])
            
            return {
                'train': pd.concat(train_dfs, ignore_index=True),
                'val': pd.concat(val_dfs, ignore_index=True),
                'test': pd.concat(test_dfs, ignore_index=True)
            }
        else:
            return self._split_single_series(df, train_ratio, val_ratio, test_ratio)
    
    def _split_single_series(self, df: pd.DataFrame, train_ratio: float,
                            val_ratio: float, test_ratio: float) -> Dict[str, pd.DataFrame]:
        """Split a single time series chronologically."""
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        return {
            'train': df.iloc[:train_end].copy(),
            'val': df.iloc[train_end:val_end].copy(),
            'test': df.iloc[val_end:].copy()
        }
    
    def scale_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None,
                      test_df: pd.DataFrame = None, feature_cols: List[str] = None) -> Tuple:
        """
        Scale features using training data statistics.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame (optional)
            test_df: Test DataFrame (optional)
            feature_cols: List of feature columns to scale
            
        Returns:
            Tuple of scaled DataFrames (train, val, test)
        """
        if feature_cols is None:
            # Exclude non-feature columns (both lowercase and capitalized versions)
            exclude_cols = ['date', 'name', 'Name', 'target', 'target_direction']
            feature_cols = [col for col in train_df.columns 
                          if col not in exclude_cols and train_df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
        
        logger.info(f"Scaling {len(feature_cols)} features using {self.scaler_type} scaler")
        
        # Fit on training data
        train_scaled = train_df.copy()
        train_scaled[feature_cols] = self.feature_scaler.fit_transform(train_df[feature_cols])
        
        # Transform validation and test data
        val_scaled = None
        test_scaled = None
        
        if val_df is not None:
            val_scaled = val_df.copy()
            val_scaled[feature_cols] = self.feature_scaler.transform(val_df[feature_cols])
        
        if test_df is not None:
            test_scaled = test_df.copy()
            test_scaled[feature_cols] = self.feature_scaler.transform(test_df[feature_cols])
        
        return train_scaled, val_scaled, test_scaled
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int,
                        target_col: str = 'close', feature_cols: List[str] = None,
                        horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            df: Input DataFrame
            sequence_length: Number of time steps to look back
            target_col: Name of the target column
            feature_cols: List of feature columns to use
            horizon: Number of steps ahead to predict
            
        Returns:
            Tuple of (X, y) arrays
        """
        if feature_cols is None:
            exclude_cols = ['date', 'name', 'Name', 'target', 'target_direction']
            feature_cols = [col for col in df.columns 
                          if col not in exclude_cols and col != target_col 
                          and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
        
        # Ensure target column is included in features
        if target_col in df.columns and target_col not in feature_cols:
            feature_cols = [target_col] + feature_cols
        
        data = df[feature_cols].values
        target_idx = feature_cols.index(target_col) if target_col in feature_cols else 0
        
        X, y = [], []
        
        for i in range(len(data) - sequence_length - horizon + 1):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length + horizon - 1, target_idx])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        
        return X, y
    
    def create_multi_horizon_sequences(self, df: pd.DataFrame, sequence_length: int,
                                      target_col: str = 'close', feature_cols: List[str] = None,
                                      horizons: List[int] = [1, 5, 30]) -> Dict:
        """
        Create sequences for multiple prediction horizons.
        
        Args:
            df: Input DataFrame
            sequence_length: Number of time steps to look back
            target_col: Name of the target column
            feature_cols: List of feature columns
            horizons: List of prediction horizons
            
        Returns:
            Dictionary with sequences for each horizon
        """
        sequences = {}
        
        for horizon in horizons:
            X, y = self.create_sequences(df, sequence_length, target_col, feature_cols, horizon)
            sequences[f'horizon_{horizon}'] = {'X': X, 'y': y}
            logger.info(f"Created sequences for {horizon}-day horizon: {X.shape[0]} samples")
        
        return sequences
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            method: Method to handle missing values ('forward_fill', 'backward_fill', 'drop', 'interpolate')
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        missing_count = df.isnull().sum().sum()
        
        if missing_count == 0:
            logger.info("No missing values found")
            return df
        
        logger.info(f"Found {missing_count} missing values. Using {method} method.")
        
        if method == 'forward_fill':
            df = df.fillna(method='ffill')
        elif method == 'backward_fill':
            df = df.fillna(method='bfill')
        elif method == 'drop':
            df = df.dropna()
        elif method == 'interpolate':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
        else:
            logger.warning(f"Unknown method: {method}. Using forward fill.")
            df = df.fillna(method='ffill')
        
        # Drop any remaining NaN values
        remaining = df.isnull().sum().sum()
        if remaining > 0:
            logger.warning(f"Dropping {remaining} remaining NaN values")
            df = df.dropna()
        
        return df
    
    def add_target_variable(self, df: pd.DataFrame, target_col: str = 'close',
                           horizon: int = 1, return_type: bool = False) -> pd.DataFrame:
        """
        Add target variable for prediction.
        
        Args:
            df: Input DataFrame
            target_col: Column to create target from
            horizon: Days ahead to predict
            return_type: If True, create target as return instead of price
            
        Returns:
            DataFrame with target column
        """
        df = df.copy()
        
        if return_type:
            # Target as percentage return
            df['target'] = df[target_col].pct_change(horizon).shift(-horizon)
            logger.info(f"Created return target for {horizon}-day ahead prediction")
        else:
            # Target as future price
            df['target'] = df[target_col].shift(-horizon)
            logger.info(f"Created price target for {horizon}-day ahead prediction")
        
        # Add direction target (binary classification)
        df['target_direction'] = (df['target'] > df[target_col]).astype(int)
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str] = None,
                       method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from the dataset.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            method: Method to detect outliers ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        df = df.copy()
        initial_len = len(df)
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'iqr':
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        elif method == 'zscore':
            for col in columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < threshold]
        
        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} outliers ({removed/initial_len*100:.2f}%)")
        
        return df
    
    def get_feature_importance_columns(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
        """
        Get list of columns suitable for feature importance analysis.
        
        Args:
            df: Input DataFrame
            exclude_cols: Columns to exclude
            
        Returns:
            List of feature column names
        """
        if exclude_cols is None:
            exclude_cols = ['date', 'name', 'Name', 'target', 'target_direction']
        
        # Only include numeric columns
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols 
                       and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
        return feature_cols
