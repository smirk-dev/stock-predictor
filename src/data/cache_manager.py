"""
Cache Manager - SQLite-based caching for stock data
Handles automatic updates, data persistence, and efficient retrieval
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class CacheManager:
    """Manage local cache for stock data using SQLite."""
    
    def __init__(self, db_path: str = None):
        """
        Initialize cache manager.
        
        Args:
            db_path: Path to SQLite database
        """
        if db_path is None:
            db_path = 'data/cache/stock_data.db'
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Stock data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, date)
                )
            ''')
            
            # Stock metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_metadata (
                    ticker TEXT PRIMARY KEY,
                    name TEXT,
                    sector TEXT,
                    industry TEXT,
                    market TEXT,
                    currency TEXT,
                    last_updated TIMESTAMP,
                    data_quality REAL
                )
            ''')
            
            # Model performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    rmse REAL,
                    mae REAL,
                    r2 REAL,
                    directional_accuracy REAL,
                    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    config TEXT
                )
            ''')
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    prediction_date TIMESTAMP,
                    target_date TEXT,
                    predicted_price REAL,
                    actual_price REAL,
                    model_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker_date ON stock_data(ticker, date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker ON stock_metadata(ticker)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_perf ON model_performance(ticker, training_date)')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def save_stock_data(self, ticker: str, df: pd.DataFrame) -> bool:
        """
        Save stock data to cache.
        
        Args:
            ticker: Stock ticker
            df: DataFrame with OHLCV data
            
        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Prepare data
                df_copy = df.copy()
                df_copy['ticker'] = ticker
                df_copy['date'] = pd.to_datetime(df_copy['date']).dt.strftime('%Y-%m-%d')
                
                # Insert or replace data
                for _, row in df_copy.iterrows():
                    conn.execute('''
                        INSERT OR REPLACE INTO stock_data 
                        (ticker, date, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ticker,
                        row['date'],
                        row.get('open'),
                        row.get('high'),
                        row.get('low'),
                        row.get('close'),
                        row.get('volume')
                    ))
                
                conn.commit()
                logger.info(f"Saved {len(df_copy)} records for {ticker}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving data for {ticker}: {e}")
            return False
    
    def get_stock_data(self, ticker: str, start_date: str = None, 
                      end_date: str = None) -> pd.DataFrame:
        """
        Retrieve stock data from cache.
        
        Args:
            ticker: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with stock data
        """
        try:
            query = "SELECT date, open, high, low, close, volume FROM stock_data WHERE ticker = ?"
            params = [ticker]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date ASC"
            
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df['name'] = ticker
                logger.info(f"Retrieved {len(df)} cached records for {ticker}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving data for {ticker}: {e}")
            return pd.DataFrame()
    
    def is_data_fresh(self, ticker: str, max_age_days: int = 1) -> bool:
        """
        Check if cached data is fresh.
        
        Args:
            ticker: Stock ticker
            max_age_days: Maximum age in days
            
        Returns:
            True if data is fresh
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT MAX(date) as latest_date 
                    FROM stock_data 
                    WHERE ticker = ?
                ''', (ticker,))
                
                result = cursor.fetchone()
                if result and result[0]:
                    latest_date = datetime.strptime(result[0], '%Y-%m-%d')
                    age = (datetime.now() - latest_date).days
                    return age <= max_age_days
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking freshness for {ticker}: {e}")
            return False
    
    def save_metadata(self, ticker: str, metadata: Dict):
        """Save stock metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO stock_metadata 
                    (ticker, name, sector, industry, market, currency, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ticker,
                    metadata.get('name'),
                    metadata.get('sector'),
                    metadata.get('industry'),
                    metadata.get('market'),
                    metadata.get('currency'),
                    datetime.now().isoformat()
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving metadata for {ticker}: {e}")
    
    def save_model_performance(self, ticker: str, model_type: str, 
                               metrics: Dict, config: Dict = None):
        """Save model performance metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO model_performance 
                    (ticker, model_type, rmse, mae, r2, directional_accuracy, config)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ticker,
                    model_type,
                    metrics.get('rmse'),
                    metrics.get('mae'),
                    metrics.get('r2'),
                    metrics.get('directional_accuracy'),
                    json.dumps(config) if config else None
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving model performance: {e}")
    
    def get_best_model(self, ticker: str) -> Optional[Dict]:
        """Get best performing model for a ticker."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT model_type, rmse, mae, r2, directional_accuracy, training_date
                    FROM model_performance
                    WHERE ticker = ?
                    ORDER BY r2 DESC, directional_accuracy DESC
                    LIMIT 1
                '''
                cursor = conn.cursor()
                cursor.execute(query, (ticker,))
                result = cursor.fetchone()
                
                if result:
                    return {
                        'model_type': result[0],
                        'rmse': result[1],
                        'mae': result[2],
                        'r2': result[3],
                        'directional_accuracy': result[4],
                        'training_date': result[5]
                    }
        except Exception as e:
            logger.error(f"Error getting best model: {e}")
        
        return None
    
    def get_cached_tickers(self) -> List[str]:
        """Get list of all cached tickers."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT DISTINCT ticker FROM stock_data ORDER BY ticker')
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting cached tickers: {e}")
            return []
    
    def clear_old_data(self, days: int = 730):
        """Clear data older than specified days."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM stock_data WHERE date < ?', (cutoff_date,))
                deleted = cursor.rowcount
                conn.commit()
                logger.info(f"Deleted {deleted} old records")
        except Exception as e:
            logger.error(f"Error clearing old data: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total records
                cursor.execute('SELECT COUNT(*) FROM stock_data')
                total_records = cursor.fetchone()[0]
                
                # Unique tickers
                cursor.execute('SELECT COUNT(DISTINCT ticker) FROM stock_data')
                unique_tickers = cursor.fetchone()[0]
                
                # Date range
                cursor.execute('SELECT MIN(date), MAX(date) FROM stock_data')
                date_range = cursor.fetchone()
                
                # Database size
                db_size = self.db_path.stat().st_size / (1024 * 1024)  # MB
                
                return {
                    'total_records': total_records,
                    'unique_tickers': unique_tickers,
                    'date_range': date_range,
                    'db_size_mb': round(db_size, 2)
                }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
