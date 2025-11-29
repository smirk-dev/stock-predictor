"""
Data Service - Unified interface for stock data with caching
Combines live data fetching with intelligent caching
"""

import pandas as pd
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import logging

from .live_data_fetcher import LiveDataFetcher
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


class DataService:
    """Unified data service with caching and live updates."""
    
    def __init__(self, cache_max_age_days: int = 1):
        """
        Initialize data service.
        
        Args:
            cache_max_age_days: Maximum cache age in days
        """
        self.fetcher = LiveDataFetcher()
        self.cache = CacheManager()
        self.cache_max_age = cache_max_age_days
    
    def get_stock_data(self, ticker: str, period: str = '2y', 
                       force_refresh: bool = False) -> Tuple[pd.DataFrame, str]:
        """
        Get stock data with intelligent caching.
        
        Args:
            ticker: Stock ticker
            period: Data period (1y, 2y, 5y, max)
            force_refresh: Force fetch from API
            
        Returns:
            Tuple of (DataFrame, source) where source is 'cache' or 'api'
        """
        # Check cache first unless force refresh
        if not force_refresh and self.cache.is_data_fresh(ticker, self.cache_max_age):
            df = self._get_from_cache(ticker, period)
            if not df.empty:
                logger.info(f"Using cached data for {ticker}")
                return df, 'cache'
        
        # Fetch from API
        logger.info(f"Fetching fresh data for {ticker}")
        df, metadata = self.fetcher.fetch_stock_data(ticker, period=period)
        
        if df.empty:
            # Fallback to cache if API fails
            logger.warning(f"API fetch failed for {ticker}, trying cache")
            df = self._get_from_cache(ticker, period)
            return df, 'cache_fallback' if not df.empty else 'error'
        
        # Save to cache
        self.cache.save_stock_data(ticker, df)
        if metadata:
            self.cache.save_metadata(ticker, metadata)
        
        return df, 'api'
    
    def _get_from_cache(self, ticker: str, period: str) -> pd.DataFrame:
        """Get data from cache with period filtering."""
        # Calculate start date based on period
        days_map = {'1y': 365, '2y': 730, '5y': 1825, 'max': 3650}
        days = days_map.get(period, 730)
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        return self.cache.get_stock_data(ticker, start_date=start_date)
    
    def search_stocks(self, query: str) -> List[Dict]:
        """
        Search for stocks by name or ticker.
        
        Args:
            query: Search query
            
        Returns:
            List of matching stocks
        """
        return self.fetcher.search_stocks(query)
    
    def get_market_overview(self) -> pd.DataFrame:
        """Get major market indices overview."""
        return self.fetcher.get_market_overview()
    
    def get_popular_stocks(self, market: str = 'us') -> List[str]:
        """
        Get popular stocks list.
        
        Args:
            market: 'us' or 'india'
            
        Returns:
            List of ticker symbols
        """
        if market.lower() == 'india':
            return self.fetcher.POPULAR_INDIAN_STOCKS
        return self.fetcher.POPULAR_US_STOCKS
    
    def validate_ticker(self, ticker: str) -> bool:
        """Validate if ticker exists."""
        return self.fetcher.validate_ticker(ticker)
    
    def get_cached_tickers(self) -> List[str]:
        """Get list of all cached tickers."""
        return self.cache.get_cached_tickers()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self.cache.get_cache_stats()
    
    def clear_old_cache(self, days: int = 730):
        """Clear cache older than specified days."""
        self.cache.clear_old_data(days)
    
    def save_model_performance(self, ticker: str, model_type: str, 
                               metrics: Dict, config: Dict = None):
        """Save model performance metrics."""
        self.cache.save_model_performance(ticker, model_type, metrics, config)
    
    def get_best_model(self, ticker: str) -> Optional[Dict]:
        """Get best performing model for a ticker."""
        return self.cache.get_best_model(ticker)
    
    def batch_download(self, tickers: List[str], period: str = '2y',
                      progress_callback=None) -> Dict[str, bool]:
        """
        Download multiple stocks in batch.
        
        Args:
            tickers: List of tickers
            period: Data period
            progress_callback: Optional callback for progress
            
        Returns:
            Dict mapping ticker to success status
        """
        results = {}
        for i, ticker in enumerate(tickers):
            try:
                df, source = self.get_stock_data(ticker, period, force_refresh=True)
                results[ticker] = not df.empty
                
                if progress_callback:
                    progress_callback(i + 1, len(tickers), ticker)
                    
            except Exception as e:
                logger.error(f"Error downloading {ticker}: {e}")
                results[ticker] = False
        
        return results
    
    def get_data_summary(self, ticker: str) -> Dict:
        """
        Get summary of available data for a ticker.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Summary dict with date range, record count, etc.
        """
        df = self.cache.get_stock_data(ticker)
        
        if df.empty:
            return {'ticker': ticker, 'available': False}
        
        return {
            'ticker': ticker,
            'available': True,
            'records': len(df),
            'start_date': df['date'].min(),
            'end_date': df['date'].max(),
            'days': (df['date'].max() - df['date'].min()).days,
            'latest_close': df.iloc[-1]['close'] if 'close' in df else None
        }
