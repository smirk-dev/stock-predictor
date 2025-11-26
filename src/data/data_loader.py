"""
Data loader for stock market data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class StockDataLoader:
    """Load and validate stock market data from CSV files."""
    
    def __init__(self, data_dir: Union[str, Path] = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Base directory containing stock data
        """
        self.data_dir = Path(data_dir) if data_dir else Path.cwd()
        
    def load_all_stocks(self, file_path: Union[str, Path] = None) -> pd.DataFrame:
        """
        Load the consolidated all stocks dataset.
        
        Args:
            file_path: Path to all_stocks_5yr.csv
            
        Returns:
            DataFrame with all stocks data
        """
        if file_path is None:
            file_path = self.data_dir / "all_stocks_5yr.csv"
        else:
            file_path = Path(file_path)
        
        logger.info(f"Loading all stocks data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            df = self._validate_and_clean(df)
            logger.info(f"Loaded {len(df)} records for {df['Name'].nunique()} stocks")
            return df
        except Exception as e:
            logger.error(f"Error loading all stocks data: {str(e)}")
            raise
    
    def load_individual_stock(self, ticker: str, stocks_dir: Union[str, Path] = None) -> pd.DataFrame:
        """
        Load data for a single stock.
        
        Args:
            ticker: Stock ticker symbol
            stocks_dir: Directory containing individual stock CSV files
            
        Returns:
            DataFrame with single stock data
        """
        if stocks_dir is None:
            stocks_dir = self.data_dir / "individual_stocks_5yr" / "individual_stocks_5yr"
        else:
            stocks_dir = Path(stocks_dir)
        
        file_path = stocks_dir / f"{ticker}_data.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found for ticker {ticker}: {file_path}")
        
        logger.info(f"Loading data for {ticker}")
        
        try:
            df = pd.read_csv(file_path)
            df = self._validate_and_clean(df)
            # Standardize column name to lowercase
            if 'Name' in df.columns:
                df.rename(columns={'Name': 'name'}, inplace=True)
            df['name'] = ticker  # Add ticker column if not present
            logger.info(f"Loaded {len(df)} records for {ticker}")
            return df
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {str(e)}")
            raise
    
    def load_multiple_stocks(self, tickers: List[str], stocks_dir: Union[str, Path] = None) -> pd.DataFrame:
        """
        Load data for multiple stocks and combine them.
        
        Args:
            tickers: List of stock ticker symbols
            stocks_dir: Directory containing individual stock CSV files
            
        Returns:
            Combined DataFrame with multiple stocks
        """
        dfs = []
        
        for ticker in tickers:
            try:
                df = self.load_individual_stock(ticker, stocks_dir)
                dfs.append(df)
            except FileNotFoundError:
                logger.warning(f"Skipping {ticker} - file not found")
                continue
        
        if not dfs:
            raise ValueError("No valid stock data could be loaded")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} total records for {len(dfs)} stocks")
        
        return combined_df
    
    def get_available_tickers(self, stocks_dir: Union[str, Path] = None) -> List[str]:
        """
        Get list of all available stock tickers.
        
        Args:
            stocks_dir: Directory containing individual stock CSV files
            
        Returns:
            List of ticker symbols
        """
        if stocks_dir is None:
            stocks_dir = self.data_dir / "individual_stocks_5yr" / "individual_stocks_5yr"
        else:
            stocks_dir = Path(stocks_dir)
        
        if not stocks_dir.exists():
            logger.error(f"Stocks directory not found: {stocks_dir}")
            return []
        
        tickers = []
        for file_path in stocks_dir.glob("*_data.csv"):
            ticker = file_path.stem.replace("_data", "")
            tickers.append(ticker)
        
        logger.info(f"Found {len(tickers)} available stock tickers")
        return sorted(tickers)
    
    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the stock data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Convert column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Ensure required columns exist
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Handle missing values
        if df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
            logger.warning("Found missing values in OHLCV data. Forward filling...")
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].fillna(method='ffill')
        
        # Remove any remaining NaN rows
        initial_len = len(df)
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        if len(df) < initial_len:
            logger.warning(f"Removed {initial_len - len(df)} rows with missing data")
        
        # Ensure positive prices and volume
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if (df[col] <= 0).any():
                logger.warning(f"Found non-positive values in {col}. Removing affected rows...")
                df = df[df[col] > 0]
        
        # Validate OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid_ohlc.any():
            logger.warning(f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships. Removing...")
            df = df[~invalid_ohlc]
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics for the dataset.
        
        Args:
            df: Stock data DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max(),
                'days': (df['date'].max() - df['date'].min()).days
            },
            'price_stats': {
                'min_close': df['close'].min(),
                'max_close': df['close'].max(),
                'mean_close': df['close'].mean(),
                'std_close': df['close'].std()
            },
            'volume_stats': {
                'min_volume': df['volume'].min(),
                'max_volume': df['volume'].max(),
                'mean_volume': df['volume'].mean()
            }
        }
        
        if 'name' in df.columns:
            summary['num_stocks'] = df['name'].nunique()
            summary['stocks'] = sorted(df['name'].unique().tolist())
        
        return summary
