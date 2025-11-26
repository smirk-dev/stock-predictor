"""
Feature engineering for stock data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create technical indicators and features for stock prediction."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        pass
    
    def add_all_features(self, df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
        """
        Add all technical indicators and features.
        
        Args:
            df: Stock data DataFrame with OHLCV data
            config: Configuration dictionary with feature parameters
            
        Returns:
            DataFrame with added features
        """
        df = df.copy()
        
        logger.info("Adding technical indicators...")
        
        # Price-based features
        df = self.add_returns(df)
        df = self.add_moving_averages(df, config)
        df = self.add_exponential_moving_averages(df, config)
        df = self.add_bollinger_bands(df)
        
        # Momentum indicators
        df = self.add_rsi(df, period=config.get('rsi_period', 14) if config else 14)
        df = self.add_macd(df, config)
        df = self.add_stochastic_oscillator(df)
        df = self.add_roc(df)
        df = self.add_williams_r(df)
        
        # Volatility indicators
        df = self.add_atr(df)
        
        # Volume indicators
        df = self.add_obv(df)
        
        # Trend indicators
        df = self.add_adx(df)
        df = self.add_cci(df)
        
        # Price patterns
        df = self.add_price_patterns(df)
        
        logger.info(f"Added {len([c for c in df.columns if c not in ['date', 'open', 'high', 'low', 'close', 'volume', 'name']])} features")
        
        return df
    
    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price returns."""
        df['daily_return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        return df
    
    def add_moving_averages(self, df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
        """Add Simple Moving Averages."""
        windows = config.get('sma_windows', [5, 10, 20, 50, 200]) if config else [5, 10, 20, 50, 200]
        
        for window in windows:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        
        # Add crossover signals
        if 20 in windows and 50 in windows:
            df['sma_20_50_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        return df
    
    def add_exponential_moving_averages(self, df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
        """Add Exponential Moving Averages."""
        windows = config.get('ema_windows', [12, 26, 50]) if config else [12, 26, 50]
        
        for window in windows:
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        return df
    
    def add_bollinger_bands(self, df: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        df['bb_middle'] = df['close'].rolling(window=window).mean()
        std = df['close'].rolling(window=window).std()
        df['bb_upper'] = df['bb_middle'] + (std * num_std)
        df['bb_lower'] = df['bb_middle'] - (std * num_std)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        return df
    
    def add_macd(self, df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        fast = config.get('macd_fast', 12) if config else 12
        slow = config.get('macd_slow', 26) if config else 26
        signal = config.get('macd_signal', 9) if config else 9
        
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    def add_stochastic_oscillator(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        return df
    
    def add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=period).mean()
        
        return df
    
    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume."""
        obv = np.where(df['close'] > df['close'].shift(1), df['volume'],
                      np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
        df['obv'] = pd.Series(obv).cumsum()
        
        return df
    
    def add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average Directional Index."""
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Calculate ATR if not already present
        if 'atr' not in df.columns:
            df = self.add_atr(df, period)
        
        pos_di = 100 * (pos_dm.rolling(window=period).mean() / df['atr'])
        neg_di = 100 * (neg_dm.rolling(window=period).mean() / df['atr'])
        
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
        df['adx'] = dx.rolling(window=period).mean()
        
        return df
    
    def add_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Commodity Channel Index."""
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        df['cci'] = (tp - sma) / (0.015 * mad)
        
        return df
    
    def add_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Williams %R."""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        
        df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)
        
        return df
    
    def add_roc(self, df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
        """Calculate Rate of Change."""
        df['roc'] = 100 * (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
        
        return df
    
    def add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern features."""
        # Higher high, lower low
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        # Gap up/down
        df['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
        df['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
        
        # Price position relative to day's range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Volume relative to average
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Add lagged features.
        
        Args:
            df: DataFrame
            columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def add_rolling_statistics(self, df: pd.DataFrame, column: str = 'close', windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Add rolling statistics."""
        for window in windows:
            df[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean()
            df[f'{column}_rolling_std_{window}'] = df[column].rolling(window=window).std()
            df[f'{column}_rolling_min_{window}'] = df[column].rolling(window=window).min()
            df[f'{column}_rolling_max_{window}'] = df[column].rolling(window=window).max()
        
        return df
