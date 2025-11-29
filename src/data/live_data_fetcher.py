"""
Live Data Fetcher - Real-time stock data from multiple sources
Supports global markets (yfinance) and Indian markets (NSE/BSE)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class LiveDataFetcher:
    """Fetch real-time stock data from multiple sources."""
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize live data fetcher.
        
        Args:
            cache_dir: Directory for caching data
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path('data/cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_stock_data(self, ticker: str, period: str = '5y', 
                        interval: str = '1d', market: str = 'auto') -> pd.DataFrame:
        """
        Fetch stock data for any ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period ('1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
            interval: Data interval ('1d', '1wk', '1mo')
            market: Market type ('us', 'india', 'auto')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Auto-detect market
            if market == 'auto':
                market = self._detect_market(ticker)
            
            if market == 'india':
                return self._fetch_indian_stock(ticker, period, interval)
            else:
                return self._fetch_global_stock(ticker, period, interval)
                
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def _fetch_global_stock(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch data using yfinance (global markets)."""
        logger.info(f"Fetching {ticker} from yfinance...")
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
            
            # Standardize column names
            df = df.reset_index()
            df.columns = df.columns.str.lower()
            
            # Rename columns to match expected format
            column_mapping = {
                'date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
            
            df = df.rename(columns=column_mapping)
            df['name'] = ticker
            
            # Select and order columns
            df = df[['date', 'name', 'open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Successfully fetched {len(df)} records for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()
    
    def _fetch_indian_stock(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch Indian stock data (NSE/BSE)."""
        # For Indian stocks, yfinance works with .NS or .BO suffix
        if not ticker.endswith('.NS') and not ticker.endswith('.BO'):
            ticker = f"{ticker}.NS"  # Default to NSE
        
        return self._fetch_global_stock(ticker, period, interval)
    
    def _detect_market(self, ticker: str) -> str:
        """Auto-detect market based on ticker format."""
        ticker_upper = ticker.upper()
        
        # Indian market indicators
        if ticker_upper.endswith('.NS') or ticker_upper.endswith('.BO'):
            return 'india'
        
        # Common Indian stock patterns (without suffix)
        indian_prefixes = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ITC', 'SBIN', 'BHARTI']
        if any(ticker_upper.startswith(prefix) for prefix in indian_prefixes):
            return 'india'
        
        return 'global'
    
    def search_stocks(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for stocks by name or ticker.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of stock information dictionaries
        """
        results = []
        query_upper = query.upper()
        
        try:
            # Try as ticker
            ticker = yf.Ticker(query_upper)
            info = ticker.info
            
            if info and 'symbol' in info:
                results.append({
                    'symbol': info.get('symbol', query_upper),
                    'name': info.get('longName', info.get('shortName', '')),
                    'exchange': info.get('exchange', ''),
                    'type': info.get('quoteType', 'EQUITY'),
                    'currency': info.get('currency', 'USD')
                })
        except:
            pass
        
        return results[:limit]
    
    def get_latest_price(self, ticker: str) -> Optional[float]:
        """Get the latest price for a ticker."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period='1d')
            if not data.empty:
                return data['Close'].iloc[-1]
        except:
            pass
        return None
    
    def get_multiple_stocks(self, tickers: List[str], period: str = '5y') -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks efficiently.
        
        Args:
            tickers: List of ticker symbols
            period: Data period
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        
        for ticker in tickers:
            df = self.fetch_stock_data(ticker, period=period)
            if not df.empty:
                results[ticker] = df
            time.sleep(0.1)  # Rate limiting
        
        return results
    
    def get_market_overview(self) -> Dict:
        """Get overview of major market indices."""
        indices = {
            'S&P 500': '^GSPC',
            'Dow Jones': '^DJI',
            'NASDAQ': '^IXIC',
            'NIFTY 50': '^NSEI',
            'SENSEX': '^BSESN'
        }
        
        overview = {}
        for name, ticker in indices.items():
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(period='2d')
                if len(data) >= 2:
                    current = data['Close'].iloc[-1]
                    previous = data['Close'].iloc[-2]
                    change = ((current - previous) / previous) * 100
                    
                    overview[name] = {
                        'value': current,
                        'change': change,
                        'ticker': ticker
                    }
            except:
                continue
        
        return overview
    
    def validate_ticker(self, ticker: str) -> bool:
        """Check if a ticker is valid."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return 'symbol' in info and info['symbol'] is not None
        except:
            return False
    
    def get_stock_info(self, ticker: str) -> Dict:
        """Get detailed stock information."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'symbol': info.get('symbol', ticker),
                'name': info.get('longName', info.get('shortName', '')),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', ''),
                'website': info.get('website', ''),
                'description': info.get('longBusinessSummary', '')
            }
        except Exception as e:
            logger.error(f"Error getting info for {ticker}: {e}")
            return {}


# Popular stock lists for quick access
POPULAR_US_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 
    'V', 'JNJ', 'WMT', 'PG', 'DIS', 'NFLX', 'PYPL', 'INTC', 'CSCO', 'PFE'
]

POPULAR_INDIAN_STOCKS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
    'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'LT.NS',
    'AXISBANK.NS', 'ICICIBANK.NS', 'WIPRO.NS', 'MARUTI.NS', 'BAJFINANCE.NS'
]
